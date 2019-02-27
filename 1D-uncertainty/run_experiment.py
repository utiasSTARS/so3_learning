import numpy as np
import torch
from torch import optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from train_and_test import *
import time
import csv
from visualize import *
import os

os.environ['OMP_NUM_THREADS'] = '4'
torch.set_num_threads(4)

def gen_train_data(N=100):
    mu, sigma = 0, 0.03
    alpha, beta = 4, 13
    w = np.random.normal(mu, sigma, N)
    x = np.concatenate((np.random.uniform(0,.6,int(N/2)), np.random.uniform(0.8,1.0,int(N/2))))
    y = x + np.sin(alpha*(x+w)) + np.sin(beta*(x+w)) + w
    #y = x**3 + w
    return (x, y)

def gen_test_data(N=100):
    x = np.linspace(-2,2, N)
    alpha, beta = 4, 13
    y = x + np.sin(alpha*x) + np.sin(beta*x)
    #y = x**3
    return (x, y)

def main():
    num_reps = 100
    minibatch_samples = 50
    train_samples = 1000
    test_samples =100
    num_epochs = 3000
    target_noise_sigma = 0.05

    use_cuda = False


    stats_list = []
    csv_header = ["Rep","Dropout-NLL", "Dropout-MSE",
    "Ensemble-NLL", "Ensemble-MSE",
    "Sigma-NLL", "Sigma-MSE",
    "HydraNet-NLL", "HydraNet-MSE",
    "HydraNet-Sigma-NLL", "HydraNet-Sigma-MSE"]

    for rep in range(num_reps):

        print('Performing repetition {}/{}...'.format(rep+1, num_reps))

        (x_train, y_train) = gen_train_data(train_samples)
        (x_test, y_test) = gen_test_data(test_samples)
        exp_data = ExperimentalData(x_train, y_train, x_test, y_test)


        visualize_data_only(x_train, y_train, x_test, y_test, filename='data.pdf')
        #return

        print('Starting dropout training')
        start = time.time()
#        (dropout_model, _) = train_nn_dropout(x_train, y_train, num_epochs=num_epochs, use_cuda=use_cuda)
        (dropout_model, _) = train_nn_dropout(x_train, y_train, minibatch_samples, num_epochs=num_epochs, use_cuda=use_cuda)
        (y_pred_dropout, sigma_pred_dropout) = test_nn_dropout(x_test, dropout_model, use_cuda=use_cuda)
        nll_dropout = compute_nll(y_test, y_pred_dropout, sigma_pred_dropout)
        mse_dropout = compute_mse(y_test, y_pred_dropout)
        print('Dropout, NLL: {:.3f} | MSE: {:.3f}'.format(nll_dropout, mse_dropout))
        visualize(x_train, y_train, x_test, y_test, y_pred_dropout, sigma_pred_dropout, nll_dropout, mse_dropout, rep,'dropout_{}.png'.format(rep))
        end = time.time()
        print('Completed in {:.3f} seconds.'.format(end - start))
#
#
        print('Starting ensemble training')

        start = time.time()
        ensemble_models = train_nn_ensemble_bootstrap(x_train, y_train, minibatch_samples, num_epochs=num_epochs, use_cuda=use_cuda, target_noise_sigma=target_noise_sigma)
        (y_pred_bs, sigma_pred_bs) = test_nn_ensemble_bootstrap(x_test, ensemble_models, use_cuda=use_cuda)
        nll_bs = compute_nll(y_test, y_pred_bs, sigma_pred_bs)
        mse_bs = compute_mse(y_test, y_pred_bs)
        print('Ensemble BS, NLL: {:.3f} | MSE: {:.3f}'.format(nll_bs, mse_bs))
        visualize(x_train, y_train, x_test, y_test, y_pred_bs, sigma_pred_bs, nll_bs, mse_bs, rep,'ensemble_{}.png'.format(rep))
        end = time.time()
        print('Completed in {:.3f} seconds.'.format(end - start))
#
#
        print('Starting sigma training')

        start = time.time()
        (sigma_model, _) = train_nn_sigma(exp_data, minibatch_samples, num_epochs=num_epochs, use_cuda=use_cuda)
        (y_pred_sigma, sigma_pred_sigma) = test_nn_sigma(x_test, sigma_model, use_cuda=use_cuda)
        nll_sigma = compute_nll(y_test, y_pred_sigma, sigma_pred_sigma)
        mse_sigma = compute_mse(y_test, y_pred_sigma)
        print('Sigma, NLL: {:.3f} | MSE: {:.3f}'.format(nll_sigma, mse_sigma))
        visualize(x_train, y_train, x_test, y_test, y_pred_sigma, sigma_pred_sigma, nll_sigma, mse_sigma, rep,'sigma_{}.png'.format(rep))
        end = time.time()
        print('Completed in {:.3f} seconds.'.format(end - start))

        print('Starting hydranet training with target_noise_sigma={}'.format(target_noise_sigma))
        #
        start = time.time()
        (hydranet_model, _) = train_hydranet(exp_data, minibatch_samples, num_heads=10, num_epochs=num_epochs, use_cuda=use_cuda, target_noise_sigma=target_noise_sigma)
        (y_pred_hydranet, sigma_pred_hydranet) = test_hydranet(x_test, hydranet_model, use_cuda=use_cuda)
        nll_hydranet = compute_nll(y_test, y_pred_hydranet, sigma_pred_hydranet)
        mse_hydranet = compute_mse(y_test, y_pred_hydranet)
        print('HydraNet, NLL: {:.3f} | MSE: {:.3f}'.format(nll_hydranet, mse_hydranet))
        visualize(x_train, y_train, x_test, y_test, y_pred_hydranet, sigma_pred_hydranet, nll_hydranet, mse_hydranet, rep,'hydranet_{}.png'.format(rep))
        end = time.time()
        print('Completed in {:.3f} seconds.'.format(end - start))

        print('Starting hydranet-sigma training')
        #
        start = time.time()
        (hydranetsigma_model, _) = train_hydranet_sigma(exp_data, minibatch_samples, num_heads=10, num_epochs=num_epochs, use_cuda=use_cuda, target_noise_sigma=target_noise_sigma)
        (y_pred_hydranetsigma, sigma_pred_hydranetsigma) = test_hydranet_sigma(x_test, hydranetsigma_model, use_cuda=use_cuda)
        nll_hydranetsigma = compute_nll(y_test, y_pred_hydranetsigma, sigma_pred_hydranetsigma)
        mse_hydranetsigma = compute_mse(y_test, y_pred_hydranetsigma)

        print('HydraNet-Sigma, NLL: {:.3f} | MSE: {:.3f}'.format(nll_hydranetsigma, mse_hydranetsigma))
        visualize(x_train, y_train, x_test, y_test, y_pred_hydranetsigma, sigma_pred_hydranetsigma, nll_hydranetsigma, mse_hydranetsigma, rep,'hydranetsigma_{}.png'.format(rep))
        end = time.time()
        print('Completed in {:.3f} seconds.'.format(end - start))

        stats_list.append([rep, nll_dropout, mse_dropout, nll_bs, mse_bs, nll_sigma, mse_sigma, nll_hydranet, mse_hydranet, nll_hydranetsigma, mse_hydranetsigma])


    csv_filename = 'figs/stats_target_noise_sigma_{}.csv'.format(target_noise_sigma)
    with open(csv_filename, "w") as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)
        writer.writerows(stats_list)


if __name__ == '__main__':
    main()
