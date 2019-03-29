#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from visualize import visualize, visualize_multiple
files = ['figs/100-run-experiment/saved_outputs/experiment_run_10_noise_0.0.pt', \
         'figs/100-run-experiment/saved_outputs/experiment_run_20_noise_0.0.pt', \
         'figs/100-run-experiment/saved_outputs/experiment_run_35_noise_0.0.pt', \
         'figs/100-run-experiment/saved_outputs/experiment_run_40_noise_0.0.pt']

x_train = []
x_test = []  
y_train = []
y_test = []
y_pred_bs = []
y_pred_hydranet = []
y_pred_sigma = []
y_pred_hydranetsigma = []
sigma_pred_bs = []
sigma_pred_hydranet = []
sigma_pred_hydranetsigma = []
sigma_pred_sigma = []
nll_bs = []
nll_hydranet = []
nll_hydranetsigma = [] 
nll_sigma = []
mse_bs = [] 
mse_hydranet = [] 
mse_hydranetsigma = []  
mse_sigma = []      
rep = []
sigma_n = []


for f in files:
    data = torch.load(f)
    x_train.append(data['x_train'])
    x_test.append(data['x_test']  )
    y_train.append(data['y_train'] )
    y_test.append(data['y_test'] )
    y_pred_bs.append(data['y_pred_bs'] )
    y_pred_hydranet.append(data['y_pred_hydranet'] )
    y_pred_sigma.append(data['y_pred_sigma'])
    y_pred_hydranetsigma.append(data['y_pred_hydranetsigma']  )
    sigma_pred_bs.append(data['sigma_pred_bs'] )
    sigma_pred_hydranet.append(data['sigma_pred_hydranet'])
    sigma_pred_hydranetsigma.append(data['sigma_pred_hydranetsigma'] )
    sigma_pred_sigma.append(data['sigma_pred_sigma'] )
    nll_bs.append(data['nll_bs']  )
    nll_hydranet.append(data['nll_hydranet']  )
    nll_hydranetsigma.append(data['nll_hydranetsigma']  )
    nll_sigma.append(data['nll_sigma'])
    mse_bs.append(data['mse_bs'] )
    mse_hydranet.append(data['mse_hydranet'])  
    mse_hydranetsigma.append(data['mse_hydranetsigma']   )
    mse_sigma.append(data['mse_sigma'] )      
    rep.append(data['rep'] )
    sigma_n.append(data['sigma_n'] )

dropout_files = ['figs/100-run-experiment/saved_outputs/dropout_experiment_run_0_noise_0.0.pt', \
         'figs/100-run-experiment/saved_outputs/dropout_experiment_run_1_noise_0.0.pt', \
         'figs/100-run-experiment/saved_outputs/dropout_experiment_run_2_noise_0.0.pt', \
         'figs/100-run-experiment/saved_outputs/dropout_experiment_run_3_noise_0.0.pt' ]

y_pred_dropout, sigma_pred_dropout, nll_dropout, mse_dropout = [], [], [], []
for f in dropout_files:
    data = torch.load(f)
    y_pred_dropout.append(data['y_pred_dropout'] )
    sigma_pred_dropout.append(data['sigma_pred_dropout'] )
    nll_dropout.append(data['nll_dropout']  )
    mse_dropout.append(data['mse_dropout']   )

plot_multiple = True
if plot_multiple:
    visualize_multiple(x_train, y_train, x_test, y_test, y_pred_bs, sigma_pred_bs, nll_bs, mse_bs, rep,'100-run-experiment/figs/supp_bagging.pdf')
    visualize_multiple(x_train, y_train, x_test, y_test, y_pred_hydranet, sigma_pred_hydranet, nll_hydranet, mse_hydranet, rep,'100-run-experiment/figs/supp_hydranet.pdf')
    visualize_multiple(x_train, y_train, x_test, y_test, y_pred_hydranetsigma, sigma_pred_hydranetsigma, nll_hydranetsigma, mse_hydranetsigma, rep,'100-run-experiment/figs/supp_hydranetsigma.pdf')
    visualize_multiple(x_train, y_train, x_test, y_test, y_pred_sigma, sigma_pred_sigma, nll_sigma, mse_sigma, rep,'100-run-experiment/figs/supp_sigma.pdf')
    visualize_multiple(x_train, y_train, x_test, y_test, y_pred_dropout, sigma_pred_dropout, nll_dropout, mse_dropout, rep,'100-run-experiment/figs/supp_dropout.pdf')    
else:
    visualize(x_train, y_train, x_test, y_test, y_pred_bs, sigma_pred_bs, nll_bs, mse_bs, rep,'100-run-experiment/figs/ensemble_{}_noise_{}.png'.format(rep, sigma_n))
    visualize(x_train, y_train, x_test, y_test, y_pred_hydranet, sigma_pred_hydranet, nll_hydranet, mse_hydranet, rep,'100-run-experiment/figs/hydranet_{}_noise_{}.png'.format(rep,sigma_n))
    visualize(x_train, y_train, x_test, y_test, y_pred_hydranetsigma, sigma_pred_hydranetsigma, nll_hydranetsigma, mse_hydranetsigma, rep,'100-run-experiment/figs/hydranetsigma_{}_noise_{}.png'.format(rep,sigma_n))
    visualize(x_train, y_train, x_test, y_test, y_pred_sigma, sigma_pred_sigma, nll_sigma, mse_sigma, rep,'100-run-experiment/figs/sigma_{}_noise_{}.png'.format(rep,sigma_n))