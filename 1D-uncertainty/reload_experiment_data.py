import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from visualize import visualize

data = torch.load('figs/noise_experiment/noise_experiment_run_0_noise_0.01.pt')
x_train = data['x_train']
x_test = data['x_test']  
y_train = data['y_train'] 
y_test = data['y_test'] 
y_pred_bs = data['y_pred_bs'] 
y_pred_hydranet = data['y_pred_hydranet'] 
y_pred_hydranetsigma = data['y_pred_hydranetsigma']  
sigma_pred_bs = data['sigma_pred_bs'] 
sigma_pred_hydranet = data['sigma_pred_hydranet']
sigma_pred_hydranetsigma = data['sigma_pred_hydranetsigma']  
nll_bs = data['nll_bs']  
nll_hydranet = data['nll_hydranet']  
nll_hydranetsigma = data['nll_hydranetsigma']  
mse_bs = data['mse_bs'] 
mse_hydranet = data['mse_hydranet']  
mse_hydranetsigma = data['mse_hydranetsigma']          
rep = data['rep'] 
sigma_n = data['sigma_n'] 

visualize(x_train, y_train, x_test, y_test, y_pred_bs, sigma_pred_bs, nll_bs, mse_bs, rep,'noise_experiment/figs/ensemble_{}_noise_{}.png'.format(rep, sigma_n))
visualize(x_train, y_train, x_test, y_test, y_pred_hydranet, sigma_pred_hydranet, nll_hydranet, mse_hydranet, rep,'noise_experiment/figs/hydranet_{}_noise_{}.png'.format(rep,sigma_n))
visualize(x_train, y_train, x_test, y_test, y_pred_hydranetsigma, sigma_pred_hydranetsigma, nll_hydranetsigma, mse_hydranetsigma, rep,'noise_experiment/figs/hydranetsigma_{}_noise_{}.png'.format(rep,sigma_n))