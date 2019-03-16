import torch, math
from lie_algebra import so3_exp
import math
from utils import *
from torchvision import transforms, datasets, models

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
        self.heads = torch.nn.ModuleList([GenericHead(D_in=self.sensor_net_dim, D_layers=512, D_out=4, dropout=False) for h in range(self.num_hydra_heads)])
        self.direct_covar_head = GenericHead(D_in=self.sensor_net_dim, D_layers=512, D_out=3, dropout=False)
            
    def forward(self, sensor_data):
        x = self.sensor_net(sensor_data)
        q_out = [normalize_vecs(head_net(x)) for head_net in self.heads]
        inv_vars = positive_fn(self.direct_covar_head(x)) + 1e-8 # Add a small non-zero number to avoid divide by zero errors
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
            Rinv_direct = I.mul(inv_vars.unsqueeze(2).expand(batch_size, 3, 3))

            if self.num_hydra_heads > 1:
                # #Convert into a concatenated tensor: N*M x D (where N=batches, M= heads)
                q_batch = q_stack.permute(1, 0, 2).contiguous().view(-1, 4)
                q_batch_mean =  q_mean.repeat([1, self.num_hydra_heads]).view(-1,4)
                phi_diff = quat_log_diff(q_batch, q_batch_mean).view(-1, self.num_hydra_heads, 3)
                Rinv = (Rinv_direct.inverse() + batch_sample_covariance(phi_diff)).inverse() #Outputs N x D - 1 x D - 1
            else:
                Rinv = Rinv_direct

            return q_mean, Rinv, Rinv_direct


def conv_unit(in_planes, out_planes, kernel_size=3, stride=1,padding=1):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding),
            torch.nn.BatchNorm2d(out_planes),
            torch.nn.ReLU()
        )


class BasicCNN(torch.nn.Module):
    def __init__(self, feature_dim=128, channels=3):
        super(BasicCNN, self).__init__()
        self.cnn0 = torch.nn.Sequential(
            # StandardBlock(D_in_sensor, self.sensor_net_dim),
            conv_unit(channels, 64, kernel_size=3, stride=2, padding=1),
            conv_unit(64, 128, kernel_size=3, stride=2, padding=1),
            conv_unit(128, 256, kernel_size=3, stride=2, padding=1),
            conv_unit(256, 512, kernel_size=3, stride=2, padding=1),
            conv_unit(512, 1024, kernel_size=3, stride=2, padding=1),
            conv_unit(1024, 1024, kernel_size=3, stride=2, padding=1),
            conv_unit(1024, 1024, kernel_size=3, stride=2, padding=1)
        )
        self.fc = torch.nn.Linear(4096, feature_dim)


    def forward(self, x):
        out = self.cnn0(x)
        out = out.view(out.shape[0], -1)
        out = self.fc(out)
        return out

class CustomResNet(torch.nn.Module):
    def __init__(self, feature_dim):
        super(CustomResNet, self).__init__()
        self.dnn = models.resnet18(pretrained=True)

        num_ftrs = self.dnn.fc.in_features
        self.dnn.fc = torch.nn.Linear(num_ftrs, feature_dim)

    def forward(self, x):
        return self.dnn(x)

    def freeze_layers(self):
        # To freeze or not to freeze...
        for param in self.dnn.parameters():
            param.requires_grad = False

        # Keep the FC layer active..
        for param in self.dnn.fc.parameters():
            param.requires_grad = True



class QuaternionCNN(torch.nn.Module):
    def __init__(self, num_hydra_heads=25, channels=2, resnet=False):
        super(QuaternionCNN, self).__init__()
        self.num_hydra_heads = num_hydra_heads

        sensor_feature_dim = 256
        if resnet:
            self.sensor_net = CustomResNet(feature_dim=sensor_feature_dim)
        else:
            self.sensor_net = BasicCNN(feature_dim=sensor_feature_dim,
                                       channels=channels)

        self.heads = torch.nn.ModuleList(
            [GenericHead(D_in=sensor_feature_dim, D_layers=256, D_out=4, dropout=True) for h in range(self.num_hydra_heads)])
        self.direct_covar_head = GenericHead(D_in=sensor_feature_dim, D_layers=256, D_out=3, dropout=False)

    def forward(self, sensor_data):
        x = self.sensor_net(sensor_data)
        q_out = [normalize_vecs(head_net(x)) for head_net in self.heads]
        inv_vars = positive_fn(self.direct_covar_head(x)) + 1e-8  # Add a small non-zero number to avoid divide by zero errors
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
            Rinv_direct = I.mul(inv_vars.unsqueeze(2).expand(batch_size, 3, 3))

            if self.num_hydra_heads > 1:
                # #Convert into a concatenated tensor: N*M x D (where N=batches, M= heads)
                q_batch = q_stack.permute(1, 0, 2).contiguous().view(-1, 4)
                q_batch_mean = q_mean.repeat([1, self.num_hydra_heads]).view(-1, 4)
                phi_diff = quat_log_diff(q_batch, q_batch_mean).view(-1, self.num_hydra_heads, 3)
                Rinv = (Rinv_direct.inverse() + batch_sample_covariance(phi_diff)).inverse()  # Outputs N x D - 1 x D - 1
            else:
                Rinv = Rinv_direct
            return q_mean, Rinv, Rinv_direct


class QuaternionDualCNN(torch.nn.Module):
    def __init__(self, num_hydra_heads=25):
        super(QuaternionDualCNN, self).__init__()
        self.num_hydra_heads = num_hydra_heads

        sensor_feature_dim = 256
        #BasicCNN(feature_dim=sensor_feature_dim) #
        self.sensor_net = CustomResNet(feature_dim=sensor_feature_dim)
        #self.sensor_net1 = self.sensor_net0#CustomResNet(feature_dim=sensor_feature_dim)

        self.heads = torch.nn.ModuleList(
            [GenericHead(D_in=2*sensor_feature_dim, D_layers=512, D_out=4, dropout=False, init_large=True) for h in range(self.num_hydra_heads)])
        self.direct_covar_head = GenericHead(D_in=2*sensor_feature_dim, D_layers=128, D_out=3, dropout=False)

    def forward(self, image_pair):
        x0 = self.sensor_net(image_pair[0])
        x1 = self.sensor_net(image_pair[1])
        x = torch.cat((x0, x1), 1)

        q_out = [normalize_vecs(head_net(x)) for head_net in self.heads]
        inv_vars = positive_fn(self.direct_covar_head(x)) + 1e-8  # Add a small non-zero number to avoid divide by zero errors
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
            Rinv_direct = I.mul(inv_vars.unsqueeze(2).expand(batch_size, 3, 3))

            if self.num_hydra_heads > 1:
                # #Convert into a concatenated tensor: N*M x D (where N=batches, M= heads)
                q_batch = q_stack.permute(1, 0, 2).contiguous().view(-1, 4)
                q_batch_mean = q_mean.repeat([1, self.num_hydra_heads]).view(-1, 4)
                phi_diff = quat_log_diff(q_batch, q_batch_mean).view(-1, self.num_hydra_heads, 3)
                Rinv = (Rinv_direct.inverse() + batch_sample_covariance(phi_diff)).inverse()  # Outputs N x D - 1 x D - 1
            else:
                Rinv = Rinv_direct

            return q_mean, Rinv, Rinv_direct


class GenericHead(torch.nn.Module):
    def __init__(self, D_in, D_out, D_layers, dropout=False, init_large=False):
        super(GenericHead, self).__init__()
        self.fc0 = torch.nn.Linear(D_in, D_layers)
        self.fc1 = torch.nn.Linear(D_layers, D_out)
        #self.bn = torch.nn.BatchNorm1d(D_layers)

        if init_large:
            self.fc0.apply(init_lin_weights)
            self.fc1.apply(init_lin_weights)
        if dropout:
            self.dropout = torch.nn.Dropout(p=0.5)
        else:
            self.dropout = None
        self.nonlin = torch.nn.PReLU()

    def forward(self, x):
        out = self.fc0(x)
        out = self.nonlin(out)
        if self.dropout:
            out = self.dropout(out)
        out = self.fc1(out)
        return out

def init_lin_weights(m):
    if type(m) == torch.nn.Linear:
        stdv = 2. / math.sqrt(m.weight.size(1))
        m.weight.data.uniform_(-stdv, stdv)

        #torch.nn.init.kaiming_normal_(m.weight)
        #m.weight.data.normal_(std=1e-2)
        #m.weight.data.fill_(0.01)
        m.bias.data.normal_(std=1e-2)

