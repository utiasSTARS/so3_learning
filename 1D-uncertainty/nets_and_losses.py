import numpy as np
import torch

#Setup and train a simple neural network
def build_NN(dropout_p, num_outputs):
    if dropout_p > 0:
        NN = torch.nn.Sequential(
            torch.nn.Linear(1, 20),
            torch.nn.SELU(),
            torch.nn.AlphaDropout(p=dropout_p),
            torch.nn.Linear(20, 20),
            torch.nn.SELU(),
            torch.nn.AlphaDropout(p=dropout_p),
            torch.nn.Linear(20, 20),
            torch.nn.SELU(),
            torch.nn.AlphaDropout(p=dropout_p),
            torch.nn.Linear(20, num_outputs),
        )
    else:
        NN = torch.nn.Sequential(
            torch.nn.Linear(1, 20),
            torch.nn.SELU(),
            torch.nn.Linear(20, 20),
            torch.nn.SELU(),
            torch.nn.Linear(20, 20),
            torch.nn.SELU(),
            torch.nn.Linear(20, num_outputs)
        )    
    return NN
    
#NN with multiple heads
def build_hydra(num_heads, num_outputs=1, direct_variance_head=False):
    class HydraHead(torch.nn.Module):
        def __init__(self, n_o):
            super(HydraHead, self).__init__()

            self.head_net = torch.nn.Sequential(
                     torch.nn.Linear(20, 20),
                     torch.nn.SELU(),
                     torch.nn.Linear(20, n_o))
            
        def forward(self, x):
            return self.head_net(x)
            
    class HydraNet(torch.nn.Module):
        def __init__(self, num_heads, num_outputs, direct_variance_head=False):
            super(HydraNet, self).__init__()
            self.shared_net = torch.nn.Sequential(
                torch.nn.Linear(1, 20),
                torch.nn.SELU(),
                torch.nn.Linear(20, 20),
                torch.nn.SELU()
            )
            #Initialize the heads
            self.num_heads = num_heads
            self.num_outputs = num_outputs
            self.heads = torch.nn.ModuleList([HydraHead(n_o=num_outputs) for h in range(num_heads)])

            if direct_variance_head:
                self.direct_variance_head = HydraHead(n_o=1)
            else:
                self.direct_variance_head = None
            
        def forward(self, x):
            y = self.shared_net(x)
            y_out = [head_net(y) for head_net in self.heads]

            #Append the direct variance to the end of the heads
            if self.direct_variance_head is not None:
                y_out.append(self.direct_variance_head(y))

            return torch.cat(y_out, 1)
            
    net = HydraNet(num_heads, num_outputs, direct_variance_head)
    return net


#NLL loss for single-headed NN
class GaussianLoss(torch.nn.Module):
    def __init__(self):
        super(GaussianLoss, self).__init__()
    
    #Based on negative log of normal distribution
    def forward(self, input, target):
        mean = input[:, 0]
        sigma2 = torch.log(1. + torch.exp(input[:, 1])) + 1e-6
        #sigma2 = torch.nn.functional.softplus(input[:, 1]) + 1e-4
        loss = torch.mean(0.5*(mean - target.squeeze())*((mean - target.squeeze())/sigma2) + 0.5*torch.log(sigma2))
        return loss

#NLL loss for HydraNet
class GaussianHydraLoss(torch.nn.Module):
    def __init__(self):
        super(GaussianHydraLoss, self).__init__()
    
    #Based on negative log of normal distribution
    def forward(self, input, target):

        mean = input[:, :-1]
        sigma2 = torch.log(1. + torch.exp(input[:, [-1]])) + 1e-6 #torch.abs(input[:, [-1]]) + 1e-6

        sigma2 = sigma2.repeat([1, mean.shape[1]])
        #sigma2 = torch.nn.functional.softplus(input[:, :, 1]) + 1e-4
        loss = 0.5*(mean - target)*((mean - target)/sigma2) + 0.5*torch.log(sigma2)
        return loss.mean()