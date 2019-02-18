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
    
class SO3IterNet(torch.nn.Module):
    def __init__(self, D_in_sensor):
        super(SO3Net, self).__init__()
        self.sensor_net_dim = D_in_sensor
        self.num_iter_heads = 1
        self.num_hydra_heads = 20

        self.sensor_net = torch.nn.Sequential(
          ResidualBlock(self.sensor_net_dim),
          ResidualBlock(self.sensor_net_dim),
          ResidualBlock(self.sensor_net_dim),
          ResidualBlock(self.sensor_net_dim),
          ResidualBlock(self.sensor_net_dim),
        )
        self.heads = torch.nn.ModuleList([IterHead(D_in=self.sensor_net_dim, D_out=3, num_hydra_heads=self.num_hydra_heads) for h in range(self.num_iter_heads)])

            
    def forward(self, sensor_data, C_op):
        y = self.sensor_net(sensor_data)
        C = C_op.clone()
        C_list = [C_op]
        for h_i, head_net in enumerate(self.heads):
            #head_in = torch.cat((y, C.view(-1, 9)), dim=1) 
            #phi = head_net(head_in)1
            phi = head_net(y)
            #print('Iter: {}. Norm: {:3.3f}'.format(h_i, phi.norm()))
            C = so3_exp(phi).bmm(C)
            #C = so3_exp(phi).bmm(C)
            C_list.append(C)
        return C, C_list


class SO3Net(torch.nn.Module):
    def __init__(self, D_in_sensor, num_hydra_heads=25):
        super(SO3Net, self).__init__()
        self.sensor_net_dim = D_in_sensor  # 256
        self.num_hydra_heads = num_hydra_heads

        self.sensor_net = torch.nn.Sequential(
            ResidualBlock(self.sensor_net_dim),
            ResidualBlock(self.sensor_net_dim),
            ResidualBlock(self.sensor_net_dim),
            ResidualBlock(self.sensor_net_dim),
            ResidualBlock(self.sensor_net_dim)
        )
        self.heads = torch.nn.ModuleList(
            [GenericHead(D_in=self.sensor_net_dim, D_out=3) for h in range(self.num_hydra_heads)])
        self.direct_covar_head = GenericHead(D_in=self.sensor_net_dim, D_out=1)

    def forward(self, sensor_data):
        x = self.sensor_net(sensor_data)
        phi_out = [head_net(x) for head_net in self.heads]
        inv_vars = torch.abs(self.direct_covar_head(x)) + 1e-8  # Add a small non-zero number to avoid divide by zero errors
        # If we are training, we just return self_heads*batch_size vectors - otherwise we apply the quat mean
        if self.training:
            phi_out = torch.cat(phi_out, 0)
            I = torch.diag(phi_out.new_ones(3)).expand(phi_out.shape[0], 3, 3)
            inv_vars = inv_vars.repeat([1, self.num_hydra_heads]).view(-1, 1, 1)
            Rinv = inv_vars * I
            return so3_exp(phi_out), Rinv
        else:
            phi_stack = torch.stack(phi_out, 0)
            phi_mean = phi_stack.mean(dim=0)
            I = torch.diag(phi_mean.new_ones(3)).expand(phi_mean.shape[0], 3, 3)
            Rinv = inv_vars.view(-1, 1, 1) * I

            if self.num_hydra_heads > 1:
                # #Convert into a concatenated tensor: N*M x D (where N=batches, M= heads)
                phi_batch = phi_stack.permute(1, 0, 2).contiguous().view(-1, self.num_hydra_heads, 3)
                #q_batch_mean = q_mean.repeat([1, self.num_hydra_heads]).view(-1, 4)
                #phi_diff = quat_log_diff(q_batch, q_batch_mean).view(-1, self.num_hydra_heads, 3)

                Rinv = (batch_sample_covariance(phi_batch)).inverse()  # Outputs N x D - 1 x D - 1

                if torch.isnan(Rinv).any():
                    raise Exception(
                        'Nans in Rinv')

            return so3_exp(phi_mean), Rinv

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
        self.direct_covar_head = GenericHead(D_in=self.sensor_net_dim, D_out=1)
            
    def forward(self, sensor_data):
        x = self.sensor_net(sensor_data)
        q_out = [normalize_vecs(head_net(x)) for head_net in self.heads]
        inv_vars = torch.abs(self.direct_covar_head(x)) + 1e-8 # Add a small non-zero number to avoid divide by zero errors
        #If we are training, we just return self_heads*batch_size vectors - otherwise we apply the quat mean
        if self.training:
            q_out = torch.cat(q_out, 0)
            I = torch.diag(q_out.new_ones(3)).expand(q_out.shape[0],3,3)
            inv_vars = inv_vars.repeat([1, self.num_hydra_heads]).view(-1,1,1) 
            Rinv = inv_vars * I
            return q_out, Rinv
        else:
            q_stack = torch.stack(q_out, 0)
            q_mean = normalize_vecs(q_stack.mean(dim=0))
            I = torch.diag(q_mean.new_ones(3)).expand(q_mean.shape[0],3,3)
            Rinv = inv_vars.view(-1,1,1) * I


            if self.num_hydra_heads > 1:
                # #Convert into a concatenated tensor: N*M x D (where N=batches, M= heads)
                q_batch = q_stack.permute(1, 0, 2).contiguous().view(-1, 4)
                q_batch_mean =  q_mean.repeat([1, self.num_hydra_heads]).view(-1,4)
                phi_diff = quat_log_diff(q_batch, q_batch_mean).view(-1, self.num_hydra_heads, 3)
                Rinv = (Rinv.inverse() + batch_sample_covariance(phi_diff)).inverse() #Outputs N x D - 1 x D - 1
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