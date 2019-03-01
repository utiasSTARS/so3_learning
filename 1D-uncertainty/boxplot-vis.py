import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import torch

sigma_n=0.05
f = pd.read_csv('figs/stats_target_noise_sigma_{}.csv'.format(sigma_n), mangle_dupe_cols=True)
noise_test = True


ensemble_nll = np.asarray(f['Ensemble-NLL']).reshape((-1,1))
ensemble_mse = np.asarray(f['Ensemble-MSE']).reshape((-1,1))
hydranet_nll = np.asarray(f['HydraNet-NLL']).reshape((-1,1))
hydranet_mse = np.asarray(f['HydraNet-MSE']).reshape((-1,1))
hydra_sigma_nll = np.asarray(f['HydraNet-Sigma-NLL']).reshape((-1,1))
hydra_sigma_mse = np.asarray(f['HydraNet-Sigma-MSE']).reshape((-1,1))

if noise_test is False:
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
    
else:
    nll_losses = np.hstack((ensemble_nll, hydranet_nll, hydra_sigma_nll))
    mse_losses = np.hstack((ensemble_mse, hydranet_mse, hydra_sigma_mse))

    plt.figure()
    plt.clf()
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.boxplot(nll_losses,vert=False)
#    plt.ylim(10^0,10^2)
    plt.title('NLL')
    plt.grid()
    plt.yticks([1, 2, 3], ['Ensemble', 'HydraNet', 'HydraNet-Sigma'])
    plt.yticks(rotation=20)
    plt.xscale('log')
    plt.savefig('figs/noise_experiment/uncertainty-NLL-noise-{}.pdf'.format(sigma_n), format='pdf', dpi=800, bbox_inches='tight')
    
    plt.figure()
    plt.clf()
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.boxplot(mse_losses,vert=False)
#    plt.ylim(0,100)
    plt.title('MSE')
    plt.grid()
    plt.yticks([1, 2, 3], ['Ensemble', 'HydraNet', 'HydraNet-Sigma'])
    plt.yticks(rotation=20)
    plt.xscale('log')
    plt.savefig('figs/noise_experiment/uncertainty-MSE-noise-{}.pdf'.format(sigma_n), format='pdf', dpi=800, bbox_inches='tight')