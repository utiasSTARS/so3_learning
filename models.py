import torch, math
from lie_algebra import so3_exp
import math
from utils import *

class StandardBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(StandardBlock, self).__init__()
        self.lin =  torch.nn.Linear(in_channels, out_channels)
        self.relu = torch.nn.ReLU()
        self.bn = torch.nn.BatchNorm1d(out_channels)
        self.dropout = torch.nn.Dropout()

    def forward(self, x):
        out = self.lin(x)
        #out = self.bn(out)
        out = self.relu(out)
        #out = self.dropout(out)
        return out

class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.lin =  torch.nn.Linear(channels, channels)
        self.relu = torch.nn.ReLU()
        self.bn = torch.nn.BatchNorm1d(channels)
        self.dropout = torch.nn.Dropout()

    def forward(self, x):
        residual = x
        out = self.lin(x)
        #out = self.bn(out)
        out = self.relu(out)
        out = out + residual
        #out = self.dropout(out)
        return out





class QuaternionNet(torch.nn.Module):
    def __init__(self, D_in_sensor, num_hydra_heads=25):
        super(QuaternionNet, self).__init__()
        self.sensor_net_dim = D_in_sensor#256
        self.num_hydra_heads = num_hydra_heads

        self.sensor_net = torch.nn.Sequential(
          #StandardBlock(D_in_sensor, self.sensor_net_dim),
          ResidualBlock(self.sensor_net_dim),
          ResidualBlock(self.sensor_net_dim),
          ResidualBlock(self.sensor_net_dim),
          ResidualBlock(self.sensor_net_dim),
          ResidualBlock(self.sensor_net_dim)
        )
        self.heads = torch.nn.ModuleList([GenericHead(D_in=self.sensor_net_dim, D_out=4) for h in range(self.num_hydra_heads)])
        self.direct_covar_head = GenericHead(D_in=self.sensor_net_dim, D_out=3)
            
    def forward(self, sensor_data):
        x = self.sensor_net(sensor_data)
        q_out = [normalize_vecs(head_net(x)) for head_net in self.heads]
        inv_vars = torch.abs(self.direct_covar_head(x)) + 1e-8 # Add a small non-zero number to avoid divide by zero errors
        #If we are training, we just return self_heads*batch_size vectors - otherwise we apply the quat mean

        if self.training:
            q_out = torch.cat(q_out, 0)
            I = torch.diag(q_out.new_ones(3)).expand(q_out.shape[0],3,3)
            inv_vars = inv_vars.repeat([self.num_hydra_heads, 1])
            Rinv = I.mul(inv_vars.unsqueeze(2).expand(q_out.shape[0], 3, 3))
            return q_out, Rinv
        else:
            q_stack = torch.stack(q_out, 0)
            q_mean = normalize_vecs(set_quat_sign(q_stack).mean(dim=0))
            batch_size = q_mean.shape[0]
            I = torch.diag(q_mean.new_ones(3)).expand(batch_size,3,3)
            Rinv = I.mul(inv_vars.unsqueeze(2).expand(batch_size, 3, 3))

            if self.num_hydra_heads > 1:
                # #Convert into a concatenated tensor: N*M x D (where N=batches, M= heads)
                q_batch = q_stack.permute(1, 0, 2).contiguous().view(-1, 4)
                q_batch_mean =  q_mean.repeat([1, self.num_hydra_heads]).view(-1,4)
                phi_diff = quat_log_diff(q_batch, q_batch_mean).view(-1, self.num_hydra_heads, 3)
                Rinv = (Rinv.inverse() + batch_sample_covariance(phi_diff)).inverse() #Outputs N x D - 1 x D - 1
            return q_mean, Rinv

def conv_unit(in_planes, out_planes, kernel_size=3, stride=1,padding=1):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding),
            torch.nn.BatchNorm2d(out_planes),
            torch.nn.PReLU()
        )


class BasicCNN(torch.nn.Module):
    def __init__(self, feature_dim=128):
        super(BasicCNN, self).__init__()
        self.cnn0 = torch.nn.Sequential(
            # StandardBlock(D_in_sensor, self.sensor_net_dim),
            conv_unit(3, 64, kernel_size=3, stride=2, padding=1),
            conv_unit(64, 64, kernel_size=3, stride=2, padding=1),
            conv_unit(64, 128, kernel_size=3, stride=2, padding=1),
            conv_unit(128, 128, kernel_size=3, stride=2, padding=1),
            conv_unit(128, 3, kernel_size=3, stride=2, padding=1),
        )
        self.fc = torch.nn.Linear(3*7*7, feature_dim)


    def forward(self, x):
        out = self.cnn0(x)
        out = out.view(out.shape[0], -1)
        out = self.fc(out)
        return out

class QuaternionCNN(torch.nn.Module):
    def __init__(self, num_hydra_heads=25):
        super(QuaternionCNN, self).__init__()
        self.num_hydra_heads = num_hydra_heads

        sensor_feature_dim = 128
        self.sensor_net = BasicCNN(feature_dim=sensor_feature_dim)

        self.heads = torch.nn.ModuleList(
            [GenericHead(D_in=sensor_feature_dim, D_out=4) for h in range(self.num_hydra_heads)])
        self.direct_covar_head = GenericHead(D_in=sensor_feature_dim, D_out=3)

    def forward(self, sensor_data):
        x = self.sensor_net(sensor_data)
        q_out = [normalize_vecs(head_net(x)) for head_net in self.heads]
        inv_vars = torch.abs(
            self.direct_covar_head(x)) + 1e-8  # Add a small non-zero number to avoid divide by zero errors
        # If we are training, we just return self_heads*batch_size vectors - otherwise we apply the quat mean

        if self.training:
            q_out = torch.cat(q_out, 0)
            I = torch.diag(q_out.new_ones(3)).expand(q_out.shape[0], 3, 3)
            inv_vars = inv_vars.repeat([self.num_hydra_heads, 1])
            Rinv = I.mul(inv_vars.unsqueeze(2).expand(q_out.shape[0], 3, 3))
            return q_out, Rinv
        else:
            q_stack = torch.stack(q_out, 0)
            q_mean = normalize_vecs(set_quat_sign(q_stack).mean(dim=0))
            batch_size = q_mean.shape[0]
            I = torch.diag(q_mean.new_ones(3)).expand(batch_size, 3, 3)
            Rinv = I.mul(inv_vars.unsqueeze(2).expand(batch_size, 3, 3))

            if self.num_hydra_heads > 1:
                # #Convert into a concatenated tensor: N*M x D (where N=batches, M= heads)
                q_batch = q_stack.permute(1, 0, 2).contiguous().view(-1, 4)
                q_batch_mean = q_mean.repeat([1, self.num_hydra_heads]).view(-1, 4)
                phi_diff = quat_log_diff(q_batch, q_batch_mean).view(-1, self.num_hydra_heads, 3)
                Rinv = (Rinv.inverse() + batch_sample_covariance(phi_diff)).inverse()  # Outputs N x D - 1 x D - 1
            return q_mean, Rinv


class IterHead(torch.nn.Module):
    def __init__(self, D_in, D_out, num_hydra_heads):
        super(IterHead, self).__init__()
        self.D_in = D_in
        self.D_out = D_out
        self.num_heads = num_hydra_heads
        self.heads = torch.nn.ModuleList([GenericHead(D_in=D_in, D_out=D_out) for h in range(num_hydra_heads)])
    def forward(self, x):
        y_out = [head_net(x) for head_net in self.heads]

        #Gather data and computer sample mean/covariance
        outputs = torch.stack(y_out, 0)
        out_mean = outputs.mean(dim=0)
        #sample_covariance = (outputs - dx_mean).mm((outputs-dx_mean).transpose(0,1))/(self.num_heads - 1)
        return out_mean



class GenericHead(torch.nn.Module):
    def __init__(self, D_in, D_out):
        super(GenericHead, self).__init__()
        self.fc0 = torch.nn.Linear(D_in, 512)
        self.fc1 = torch.nn.Linear(512, D_out)
        #self.fc0.apply(init_lin_weights)
        # self.fc1.apply(init_lin_weights)
        #self.dropout = torch.nn.Dropout()
        self.nonlin = torch.nn.PReLU()

    def forward(self, x):
        out = self.fc0(x)
        out = self.nonlin(out)
        # out = self.dropout(out)
        out = self.fc1(out)
        return out

def init_lin_weights(m):
    if type(m) == torch.nn.Linear:
        stdv = 2. / math.sqrt(m.weight.size(1))
        m.weight.data.uniform_(-stdv, stdv)

        #torch.nn.init.kaiming_normal_(m.weight)
        #m.weight.data.normal_(std=1e-2)
        #m.weight.data.fill_(0.01)
        m.bias.data.fill_(0.)