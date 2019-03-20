import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import torch
from visualize import boxplot

def create_noise_plots():
    sigma_n = [0, 0.01, 0.05, 0.25]
    models = ['Ensemble', 'HydraNet', 'HydraNet-Sigma']
    nll_losses = []
    mse_losses = []

    for m in models:
        nll = []
        mse = []
        for n in sigma_n:
            f = pd.read_csv('figs/stats_target_noise_sigma_{}.csv'.format(n), mangle_dupe_cols=True)
            nll.append(np.asarray(f[m + '-NLL']).reshape((-1, 1)))
            mse.append(np.asarray(f[m + '-MSE']).reshape((-1, 1)))
        nll_losses.append(np.hstack((nll)))
        mse_losses.append(np.hstack((mse)))

    positions = [[1, 2, 3, 4], [6, 7, 8, 9], [11, 12, 13, 14]]

    boxplot(nll_losses, title='NLL', legend=models, positions=positions,
            save_path='figs/noise_experiment/uncertainty-NLL-noise-{}.pdf'.format(sigma_n))
    boxplot(mse_losses, title='MSE', legend=models, positions=positions,
            save_path='figs/noise_experiment/uncertainty-MSE-noise-{}.pdf'.format(sigma_n))




def create_mse_plot():
    f = pd.read_csv('figs/100-run-experiment/stats.csv', mangle_dupe_cols=True)
    ensemble_mse = np.asarray(f['Ensemble-MSE']).reshape((-1, 1))
    hydranet_mse = np.asarray(f['HydraNet-MSE']).reshape((-1, 1))
    hydranet_sigma_mse = np.asarray(f['HydraNet-Sigma-MSE']).reshape((-1, 1))
    sigma_mse = np.asarray(f['Sigma-MSE']).reshape((-1, 1))
    dropout_mse = np.asarray(f['Dropout-MSE']).reshape((-1, 1))
    mse_losses = np.hstack((dropout_mse, ensemble_mse, sigma_mse, hydranet_mse, hydranet_sigma_mse))

    font_size = 14
    plt.figure()
    plt.clf()
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.boxplot(mse_losses)
    #plt.ylim(0,100)
    plt.title('MSE')
    plt.grid()
    plt.xticks([1, 2, 3, 4, 5], ['Dropout', 'Ensemble', 'Sigma', 'HydraNet \n(no $\sigma_{direct}$)', 'HydraNet'], labelsize=font_size)
    plt.xticks(rotation=25)

    plt.yscale('log')
    plt.savefig('figs/uncertainty-MSE.pdf', format='pdf', dpi=800, bbox_inches='tight')




def create_nll_plot():
    f = pd.read_csv('figs/100-run-experiment/stats.csv', mangle_dupe_cols=True)
    ensemble_nll = np.asarray(f['Ensemble-NLL']).reshape((-1,1))
    hydranet_nll = np.asarray(f['HydraNet-NLL']).reshape((-1,1))
    hydranet_sigma_nll = np.asarray(f['HydraNet-Sigma-NLL']).reshape((-1,1))
    sigma_nll = np.asarray(f['Sigma-NLL']).reshape((-1,1))
    dropout_nll = np.asarray(f['Dropout-NLL']).reshape((-1,1))
    nll_losses = np.hstack((dropout_nll, ensemble_nll, sigma_nll, hydranet_nll, hydranet_sigma_nll))

    font_size = 8

    plt.figure(figsize=(3,1.5))
    plt.clf()

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.boxplot(nll_losses)
    #plt.ylim(0,100)
    #plt.title('NLL')
    plt.grid()
    plt.xticks([1, 2, 3, 4, 5], ['Dropout', 'Ensemble', '$\sigma_d$', 'HydraNet', 'HydraNet\ \n(with $\sigma_d$)'], fontsize=font_size-2)
    #plt.tick_params(axis='both', which='major', labelsize=font_size)

    plt.xticks(rotation=25)
    plt.yscale('log')
    plt.gca().tick_params(axis='y', labelsize=font_size)

    plt.savefig('figs/uncertainty-NLL.pdf', format='pdf', bbox_inches='tight')

def main():
    create_nll_plot()

if __name__ == '__main__':
    main()