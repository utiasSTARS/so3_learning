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
#from train_test import *
from train_test_quat import *

from loaders import PlanetariumData
from torch.utils.data import Dataset, DataLoader



from utils import AverageMeter, compute_normalization

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
    args = parser.parse_args()
    
    #train_dataset_path = 'data/dataset3_uvd_absobs.mat'
    #valid_dataset_path = 'data/dataset3_uvd_absobs.mat'

    # range_all = list(range(0, 1900))
    # random.shuffle(range_all)
    # k_range_train = range_all[0:1500]
    # k_range_valid = range_all[1500:1900]

    train_dataset_path = 'simulation/orbital/train_abs.mat'
    valid_dataset_path = 'simulation/orbital/valid_abs.mat'

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


    num_hydra_heads=1
    model = QuaternionNet(D_in_sensor=train_dataset['y_k_j'].shape[0]*train_dataset['y_k_j'].shape[2], num_hydra_heads=num_hydra_heads)

    # model = SO3Net(D_in_sensor=train_dataset['y_k_j'].shape[0] * train_dataset['y_k_j'].shape[2],
    #                       num_hydra_heads=num_hydra_heads)
    model.to(dtype=tensor_type, device=device)

    #pretrained_model = torch.load('sensor_net_pretrained.pt')
    #model.sensor_net.load_state_dict(pretrained_model['sensor_net'])
    #model.direct_covar_head.load_state_dict(pretrained_model['direct_covar_head'])

    loss_fn.to(dtype=tensor_type, device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    #Load datasets
    normalization = compute_normalization(train_dataset).to(device)

    train_loader = DataLoader(PlanetariumData(train_dataset, k_range=k_range_train, normalization=normalization, mat_targets=False),
                        batch_size=args.batch_size, pin_memory=True, 
                        shuffle=True, num_workers=1, drop_last=False)
    valid_loader = DataLoader(PlanetariumData(valid_dataset, k_range=k_range_valid,normalization=normalization, mat_targets=False),
                        batch_size=args.batch_size, pin_memory=True,
                        shuffle=False, num_workers=1, drop_last=False)
    total_time = 0.
    now = datetime.datetime.now()
    start_datetime_str = '{}-{}-{}-{}-{}-{}'.format(now.year, now.month, now.day, now.hour, now.minute, now.second)


    #Configuration
    config = {
        'device': device
    }
    epoch_time = AverageMeter()
    avg_train_loss, train_ang_error, train_nll = validate(model, train_loader, loss_fn, config)
    avg_valid_loss, valid_ang_error, valid_nll = validate(model, valid_loader, loss_fn, config, epoch=0, output_sigma_plot=True)
    print('Starting Training \t' 
          'Train (Err/NLL) | Valid (Err/NLL) {:3.3f} / {:3.3f} | {:.3f} / {:3.3f}\t'.format(
            train_ang_error, train_nll, valid_ang_error, valid_nll))

    best_valid_nll = valid_nll
    for epoch in range(args.total_epochs):
        end = time.time()
        avg_train_loss = train(model, train_loader, loss_fn, optimizer, config)

        output_sigma_plot = (epoch%1==0)
        _, train_ang_error, train_nll = validate(model, train_loader, loss_fn, config)
        avg_valid_loss, valid_ang_error, valid_nll = validate(model, valid_loader, loss_fn, config, epoch=epoch+1, output_sigma_plot=output_sigma_plot)

        # Measure elapsed time
        epoch_time.update(time.time() - end)


        if valid_nll < best_valid_nll:
            torch.save({
                'model': model.state_dict(),
            }, 'simulation/saved_plots/best_model_heads_{}_epoch_{}.pt'.format(model.num_hydra_heads, epoch+1))

        if epoch%args.epoch_display == 0:     
            print('Epoch {}. Loss (Train/Valid) {:.3E} / {:.3E} \t'
            'Train (Err/NLL) | Valid (Err/NLL) {:3.3f} / {:3.3f} | {:.3f} / {:3.3f}\t'
            'Epoch Time {epoch_time.val:.3f} (avg: {epoch_time.avg:.3f})'.format(
                epoch+1, avg_train_loss, avg_valid_loss, train_ang_error, train_nll, valid_ang_error, valid_nll, epoch_time=epoch_time))

