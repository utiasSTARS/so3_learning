import numpy as np
import torch
from torch.autograd import Variable
from nets_and_losses import *
from visualize import *
import os

os.environ['OMP_NUM_THREADS'] = '1'
torch.set_num_threads(1)

class ExperimentalData():
    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
    
        

def compute_nll(y_true, y_pred, sigma_pred):
    sigma2 = sigma_pred**2
    losses = (0.5/(sigma2))*(y_pred - y_true)**2 + 0.5*np.log(sigma2) + 0.5*np.log(2*np.pi)
    return np.mean(losses)

def compute_mse(y_true, y_pred):
    mse = np.mean((y_pred - y_true)**2)
    return mse


#Setup a generic training function
def train_minibatch(model, loss, optimizer, x_val, y_val):
    
    x = Variable(x_val)
    y = Variable(y_val)

    # Reset gradient
    optimizer.zero_grad()

    # Forward
    fx = model.forward(x)

    output = loss.forward(fx, y)

    # Backward
    output.backward()

    # Update parameters
    optimizer.step()

#    return output.data[0]
    return output.item() #fixed for new pytorch versionx

def np_to_torch(x, y):
    return (torch.from_numpy(x).float().view(-1, 1), 
            torch.from_numpy(y).float().view(-1, 1)) #Ensure that the variables are Nx1

def train_nn_dropout(x_train, y_train, batch_size, num_epochs=5000, use_cuda=True):
    (x_train, y_train) = np_to_torch(x_train, y_train)
#    x_test = exp_data.x_test
#    y_test = exp_data.y_test

    N = x_train.shape[0]
    dropout_p = 0.01
    l2 = 5e-8
    tau_inv = 0.01
    l2_decay = l2 * (1 - dropout_p)* tau_inv / (2 * N)
    N = x_train.shape[0]

    model = build_NN(dropout_p, num_outputs=1)
    loss = torch.nn.MSELoss()

    if use_cuda:
        print('Using CUDA...')
        x_train = x_train.cuda()
        y_train = y_train.cuda()
        model.cuda()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5, weight_decay=l2_decay)

    for e in range(num_epochs):
        num_batches = N // batch_size
        cost = 0.
        for k in range(num_batches):
            start, end = k * batch_size, (k + 1) * batch_size
            cost += (1/num_batches)*train_minibatch(model, loss, optimizer, x_train[start:end], y_train[start:end])
        
        if e%100 == 0:
            print('Epoch: {}/{}. Cost: {:.3f}.'.format(e, num_epochs, cost))

    return (model, cost)

def test_nn_dropout(x_test, model, stoch_passes=50, use_cuda=True):
    
    num_test_samples = x_test.shape[0]
    x_t = np.broadcast_to(x_test, (stoch_passes, num_test_samples)).T.flatten()

    if use_cuda:
        x_t = Variable(torch.from_numpy(x_t).float().view(-1,1).cuda(), volatile=True)
    else:
        x_t = Variable(torch.from_numpy(x_t).float().view(-1,1), volatile=True)

    #Evaluate the model
    y_t = model.forward(x_t).data.view(-1).cpu().numpy()
    
    #Compute empirical means for each x_t pass
    y_t = y_t.reshape((num_test_samples, stoch_passes))

    y_t_mean = np.mean(y_t, axis=1)
    y_t_sigma = np.sqrt(np.var(y_t, axis=1, ddof=1))

    return (y_t_mean, y_t_sigma)



def train_nn_ensemble_bootstrap(x_train, y_train, batch_size, num_models=10,num_epochs=5000, use_cuda=True, target_noise_sigma=0.):
    (x_train, y_train) = np_to_torch(x_train, y_train)
#    x_test = exp_data.x_test
#    y_test = exp_data.y_test

    #Necessary to have the same noise at every epoch
    torch.manual_seed(42)

    N = x_train.shape[0]
    model_list = []

    if use_cuda:
        print('Using CUDA...')
        x_train = x_train.cuda()
        y_train = y_train.cuda()
  

    for m_i in range(num_models):
        #Sample with replacement
        indices = torch.from_numpy(np.random.choice(N, N, replace=True))
        if use_cuda:
            indices = indices.cuda()

        x_train_i = x_train[indices]
        y_train_i = y_train[indices] + target_noise_sigma*torch.randn_like(y_train[indices])
        
        #Do not use dropout
        model = build_NN(dropout_p=0, num_outputs=1)
        if use_cuda:
            model.cuda()

        loss = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

        for e in range(num_epochs):
            num_batches = N // batch_size
            cost = 0.
            for k in range(num_batches):
                start, end = k * batch_size, (k + 1) * batch_size
                cost += (1/num_batches)*train_minibatch(model, loss, optimizer, x_train_i[start:end], y_train_i[start:end])
            if e%100 == 0:
                print('Epoch: {}/{}. Cost: {:.3f}.'.format(e, num_epochs, cost))

        model_list.append(model)
   
    return(model_list)


def test_nn_ensemble_bootstrap(x_test, model_list, use_cuda=True):

    if use_cuda:
        x_t = Variable(torch.from_numpy(x_test).float().view(-1,1).cuda(), volatile=True)
    else:
        x_t = Variable(torch.from_numpy(x_test).float().view(-1,1), volatile=True)

    y_t = np.zeros((len(model_list), x_test.shape[0]))

    for m_i in range(len(model_list)):
        #Evaluate the model
        model_list[m_i].eval()

        y_t[m_i] = model_list[m_i].forward(x_t).data.view(-1).cpu().numpy()

    y_t_mean = np.mean(y_t, axis=0)
    y_t_sigma = np.sqrt(np.var(y_t, axis=0, ddof=1))

    return (y_t_mean, y_t_sigma)



def train_hydranet(exp_data, batch_size, num_heads=10, num_epochs=5000, use_cuda=True, target_noise_sigma=0.):
    (x_train, y_train) = np_to_torch(exp_data.x_train, exp_data.y_train)
    x_test = exp_data.x_test
    y_test = exp_data.y_test

    #Necessary to have the same noise at every epoch
    torch.manual_seed(42)

    model = build_hydra(num_heads)
    loss = torch.nn.MSELoss()

    if use_cuda:
        print('Using CUDA...')
        x_train = x_train.cuda()
        y_train = y_train.cuda()
        model.cuda()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    #Duplicate the targets for each of the 'heads'
    y_train_hydra =  y_train.squeeze(1).repeat(num_heads,1).t()
    y_train_hydra = y_train_hydra + target_noise_sigma*torch.randn_like(y_train_hydra)

    N = x_train.shape[0]

    for e in range(num_epochs):
        num_batches = N // batch_size
        cost = 0.
        for k in range(num_batches):
            start, end = k * batch_size, (k + 1) * batch_size
            cost += (1/num_batches)*train_minibatch(model, loss, optimizer, x_train[start:end], y_train_hydra[start:end])
        if e%100 == 0:
            print('Epoch: {}/{}. Cost: {:.3f}.'.format(e, num_epochs, cost))
    return (model, cost)


def test_hydranet(x_test, model, use_cuda=True):

    
    num_test_samples = x_test.shape[0]

    if use_cuda:
        x_t = Variable(torch.from_numpy(x_test).float().view(-1,1).cuda(), volatile=True)
    else:
        x_t = Variable(torch.from_numpy(x_test).float().view(-1,1), volatile=True)
    
    #Evaluate the model
    model.eval()
    y_t = model.forward(x_t).data.view(-1).cpu().numpy()
    
    #Compute empirical means for each head
    y_t = y_t.reshape((num_test_samples, model.num_heads))
    
    y_t_mean = np.mean(y_t, axis=1)
    y_t_sigma = np.sqrt(np.var(y_t, axis=1,ddof=1))

    return (y_t_mean, y_t_sigma)

def train_nn_sigma(exp_data, batch_size, num_epochs=5000, use_cuda=True):
    (x_train, y_train) = np_to_torch(exp_data.x_train, exp_data.y_train)
    x_test = exp_data.x_test
    y_test = exp_data.y_test
    model = build_NN(dropout_p=0, num_outputs=2)
    loss = GaussianLoss()

    if use_cuda:
        print('Using CUDA...')
        x_train = x_train.cuda()
        y_train = y_train.cuda()
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    N = x_train.shape[0]

    for e in range(num_epochs):
        num_batches = N // batch_size
        cost = 0.
        for k in range(num_batches):
            start, end = k * batch_size, (k + 1) * batch_size
            cost += (1/num_batches)*train_minibatch(model, loss, optimizer, x_train[start:end], y_train[start:end])
        if e%100 == 0:
            print('Epoch: {}/{}. Cost: {:.3f}.'.format(e, num_epochs, cost))
    return (model, cost)

  
def test_nn_sigma(x_test, model, use_cuda=True):
    num_test_samples = x_test.shape[0]

    if use_cuda:
        x_t = Variable(torch.from_numpy(x_test).float().view(-1,1).cuda(), volatile=True)
    else:
        x_t = Variable(torch.from_numpy(x_test).float().view(-1,1), volatile=True)    
    
    #Evaluate the model
    model.eval()
    y_t = model.forward(x_t).data.cpu().numpy()
    
    #Compute empirical means for each x_t pass
    y_t_mean = y_t[:, 0]
    y_t_sigma2 = np.log(1. + np.exp(y_t[:, 1])) + 1e-6

    return (y_t_mean, np.sqrt(y_t_sigma2))


def train_hydranet_sigma(exp_data, batch_size, num_heads=10, num_epochs=5000, use_cuda=True, target_noise_sigma=0.):
    (x_train, y_train) = np_to_torch(exp_data.x_train, exp_data.y_train)
    x_test = exp_data.x_test
    y_test = exp_data.y_test

    model = build_hydra(num_heads, num_outputs=1, direct_variance_head=True)
    loss = GaussianHydraLoss()

    #Necessary to have the same noise at every epoch
    torch.manual_seed(42)

    if use_cuda:
        print('Using CUDA...')
        x_train = x_train.cuda()
        y_train = y_train.cuda()
        model.cuda()
        
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.1)
    #optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    #optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    #Duplicate the targets for each of the 'heads'
    y_train_hydra =  y_train.squeeze(1).repeat(num_heads,1).t()
    y_train_hydra = y_train_hydra + target_noise_sigma*torch.randn_like(y_train_hydra)

    N = x_train.shape[0]

    for e in range(num_epochs):
        num_batches = N // batch_size
        cost = 0.
        for k in range(num_batches):
            start, end = k * batch_size, (k + 1) * batch_size
            cost += (1/num_batches)*train_minibatch(model, loss, optimizer, x_train[start:end], y_train_hydra[start:end])
        if e%100 == 0:
            print('Epoch: {}/{}. Cost: {:.3f}.'.format(e, num_epochs, cost))

    return (model, cost)

def test_hydranet_sigma(x_test, model, use_cuda=True):
    num_test_samples = x_test.shape[0]
    if use_cuda:
        x_t = Variable(torch.from_numpy(x_test).float().view(-1,1).cuda(), volatile=True)
    else:
        x_t = Variable(torch.from_numpy(x_test).float().view(-1,1), volatile=True)    
    
    #Evaluate the model
    model.eval()
    y_t = model.forward(x_t).data.cpu().numpy()
    n = model.num_heads
    
    mu_i = y_t[:,:-1] # outputs from the n heads
    mu_star = np.mean(mu_i, axis=1) #The means of the n hydra head predictions
    
    sigma2_i = np.log(1. + np.exp(y_t[:,-1])) + 1e-6 #last head gives sigma2_i
    sigma2_hydra = np.var(mu_i, axis=1, ddof=1) #variance of hydranet outputs
    sigma2_star = sigma2_i + sigma2_hydra
    sigma_star = np.sqrt(sigma2_star)

    return (mu_star, sigma_star)
