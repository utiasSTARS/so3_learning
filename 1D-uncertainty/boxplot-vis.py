import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import torch
from visualize import boxplot

noise_test = True
sigma_n=[0,0.01,0.05]
models = ['Ensemble', 'HydraNet', 'HydraNet-Sigma']
nll_losses = []
mse_losses = []

if noise_test:
    for m in models:
        nll=[]
        mse=[]
        for n in sigma_n:
            f = pd.read_csv('figs/stats_target_noise_sigma_{}.csv'.format(n), mangle_dupe_cols=True)
            nll.append(np.asarray(f[m+'-NLL']).reshape((-1,1)))
            mse.append(np.asarray(f[m+'-MSE']).reshape((-1,1)))
        nll_losses.append(np.hstack((nll)))
        mse_losses.append(np.hstack((mse)))

    positions = [[1,2,3], [5,6,7], [9,10,11]]
    
    boxplot(nll_losses, title='NLL',legend=models, positions=positions, save_path='figs/noise_experiment/uncertainty-NLL-noise-{}.pdf'.format(sigma_n))
    boxplot(mse_losses, title='MSE', legend=models, positions=positions, save_path='figs/noise_experiment/uncertainty-MSE-noise-{}.pdf'.format(sigma_n))
    
else:
    sigma_nll = np.asarray(f['Sigma-NLL']).reshape((-1,1))
    sigma_mse = np.asarray(f['Sigma-MSE']).reshape((-1,1))
    dropout_nll = np.asarray(f['Dropout-NLL']).reshape((-1,1))
    dropout_mse = np.asarray(f['Dropout-MSE']).reshape((-1,1))
    nll_losses = np.hstack((dropout_nll, ensemble_nll, sigma_nll, hydranet_nll, hydra_sigma_nll))
    mse_losses = np.hstack((dropout_mse, ensemble_mse, sigma_mse, hydranet_mse, hydra_sigma_mse))
    plt.figure()
    plt.clf()
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.boxplot(nll_losses)
    #plt.ylim(0,100)
    plt.title('NLL')
    plt.grid()
    plt.xticks([1, 2, 3, 4, 5], ['Dropout', 'Ensemble', 'Sigma', 'HydraNet', 'HydraNet-Sigma'])
    plt.xticks(rotation=20)
    plt.yscale('log')
    plt.savefig('figs/uncertainty-NLL.pdf', format='pdf', dpi=800, bbox_inches='tight')
    
    plt.figure()
    plt.clf()
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.boxplot(mse_losses)
    #plt.ylim(0,100)
    plt.title('MSE')
    plt.grid()
    plt.xticks([1, 2, 3, 4, 5], ['Dropout', 'Ensemble', 'Sigma', 'HydraNet', 'HydraNet-Sigma'])
    plt.xticks(rotation=20)
    plt.yscale('log')
    plt.savefig('figs/uncertainty-MSE.pdf', format='pdf', dpi=800, bbox_inches='tight')