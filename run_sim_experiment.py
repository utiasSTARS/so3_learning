import numpy as np
import torch
from torch import autograd
import os


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.io as sio
import math
from models import *
from loss import *
from liegroups.torch import SE3
import time, sys
import argparse
import random
import datetime
from train_test import *
from loaders import PlanetariumData
from torch.utils.data import Dataset, DataLoader
from utils import AverageMeter, compute_normalization
from vis import *

if __name__ == '__main__':
    #Reproducibility
    #torch.manual_seed(7)
    #random.seed(72)

    parser = argparse.ArgumentParser(description='3D training arguments.')
    parser.add_argument('--cuda', action='store_true', default=False)                
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epoch_display', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--total_epochs', type=int, default=100)
    parser.add_argument('--q_target_sigma', type=float, default=0.05)
    parser.add_argument('--num_heads', type=int, default=25)

    args = parser.parse_args()
    print(args)
    train_dataset_path = 'simulation/orbital/train_abs.mat'
    valid_dataset_path = 'simulation/orbital/valid_abs_ood.mat'

    k_range_train = range(0, 15000)
    k_range_valid = range(0, 500)


    train_dataset = sio.loadmat(train_dataset_path)
    valid_dataset = sio.loadmat(valid_dataset_path)

    #loss_fn = SO3FrobNorm()
    #loss_fn = QuatLoss()
    loss_fn = QuatNLLLoss()
    #loss_fn = SO3NLLLoss()

    #Float or Double?
    tensor_type = torch.float
    device = torch.device('cuda:0') if args.cuda else torch.device('cpu')


    model = QuaternionNet(D_in_sensor=train_dataset['y_k_j'].shape[0]*train_dataset['y_k_j'].shape[2], num_hydra_heads=args.num_heads)

    # model = SO3Net(D_in_sensor=train_dataset['y_k_j'].shape[0] * train_dataset['y_k_j'].shape[2],
    #                       num_hydra_heads=num_hydra_heads)
    model.to(dtype=tensor_type, device=device)

    # pretrained_model = torch.load('simulation/saved_plots/best_model_heads_1_epoch_74.pt')
    #     # model.sensor_net.load_state_dict(pretrained_model['sensor_net'])
    #     # model.direct_covar_head.load_state_dict(pretrained_model['direct_covar_head'])

    loss_fn.to(dtype=tensor_type, device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    #Load datasets
    normalization = compute_normalization(train_dataset).to(device)

    train_loader = DataLoader(PlanetariumData(train_dataset, k_range=k_range_train, normalization=normalization, mat_targets=False),
                        batch_size=args.batch_size, pin_memory=True, 
                        shuffle=True, num_workers=8, drop_last=False)
    valid_loader = DataLoader(PlanetariumData(valid_dataset, k_range=k_range_valid,normalization=normalization, mat_targets=False),
                        batch_size=args.batch_size, pin_memory=True,
                        shuffle=False, num_workers=8, drop_last=False)
    total_time = 0.
    now = datetime.datetime.now()
    start_datetime_str = '{}-{}-{}-{}-{}-{}'.format(now.year, now.month, now.day, now.hour, now.minute, now.second)


    #Configuration
    config = {
        'device': device
    }
    epoch_time = AverageMeter()
    avg_train_loss, train_ang_error, train_nll = validate(model, train_loader, loss_fn, config)
    avg_valid_loss, valid_ang_error, valid_nll, predict_history = validate(model, valid_loader, loss_fn, config, output_history=True)

    #Visualize
    sigma_filename = 'simulation/saved_plots/sigma_plot_heads_{}_epoch_{}.pdf'.format(model.num_hydra_heads, 0)
    plot_errors_with_sigmas(predict_history[0], predict_history[1], predict_history[2], predict_history[3], filename=sigma_filename)

    print('Starting Training \t' 
          'Train (Err/NLL) | Valid (Err/NLL) {:3.3f} / {:3.3f} | {:.3f} / {:3.3f}\t'.format(
            train_ang_error, train_nll, valid_ang_error, valid_nll))

    best_valid_nll = valid_nll
    for epoch in range(args.total_epochs):
        end = time.time()
        avg_train_loss = train(model, train_loader, loss_fn, optimizer, config, q_target_sigma=args.q_target_sigma)

        _, train_ang_error, train_nll = validate(model, train_loader, loss_fn, config)
        avg_valid_loss, valid_ang_error, valid_nll, predict_history = validate(model, valid_loader, loss_fn, config, output_history=True)

        # Measure elapsed time
        epoch_time.update(time.time() - end)


        if valid_nll < best_valid_nll:
            print('New best validation angular error! Outputting plots and saving model.')

            best_valid_nll = valid_nll
            sigma_filename = 'simulation/saved_plots/sigma_plot_heads_{}_epoch_{}.pdf'.format(model.num_hydra_heads, epoch+1)
            nees_filename = 'simulation/saved_plots/nees_plot_heads_{}_epoch_{}.pdf'.format(model.num_hydra_heads, epoch+1)

            plot_errors_with_sigmas(predict_history[0], predict_history[1], predict_history[2], predict_history[3], filename=sigma_filename)
            plot_nees(predict_history[0], predict_history[1], predict_history[2], filename=nees_filename)

            abs_filename = 'simulation/saved_plots/abs_sigma_plot_heads_{}_epoch_{}.pdf'.format(model.num_hydra_heads, epoch + 1)
            plot_abs_with_sigmas(predict_history[0], predict_history[1], predict_history[2], predict_history[3],
                                    filename=abs_filename)
            torch.save({
                'full_model': model.state_dict(),
                'sensor_net': model.sensor_net.state_dict(),
                'direct_covar_head': model.direct_covar_head.state_dict(),
                'predict_history': predict_history,
                'epoch': epoch + 1,
            }, 'simulation/saved_plots/best_model_heads_{}_epoch_{}.pt'.format(model.num_hydra_heads, epoch + 1))

        if epoch%args.epoch_display == 0:     
            print('Epoch {}. Loss (Train/Valid) {:.3E} / {:.3E} \t'
            'Train (Err/NLL) | Valid (Err/NLL) {:3.3f} / {:3.3f} | {:.3f} / {:3.3f}\t'
            'Epoch Time {epoch_time.val:.3f} (avg: {epoch_time.avg:.3f})'.format(
                epoch+1, avg_train_loss, avg_valid_loss, train_ang_error, train_nll, valid_ang_error, valid_nll, epoch_time=epoch_time))

