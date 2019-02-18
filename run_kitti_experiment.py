import numpy as np
import torch
from torch import autograd
import os


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.io as sio
import math
from models import SO3Net, QuaternionNet
from loss import SO3FrobNorm, QuatLoss
from liegroups.torch import SE3
import time, sys
import argparse
import random
import datetime
#from train_test import *
from train_test_quat import *

#Reproducibility
torch.manual_seed(7)
random.seed(7)

from utils import AverageMeter, compute_normalization

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='3D training arguments.')
    parser.add_argument('--cuda', action='store_true', default=False)                
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epoch_display', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--total_epochs', type=int, default=500)
    args = parser.parse_args()
    
    train_dataset_path = 'data/dataset3_uvd_absobs.mat'
    valid_dataset_path = 'data/dataset3_uvd_absobs.mat'

    # range_all = list(range(0, 1900))
    # random.shuffle(range_all)
    # k_range_train = range_all[0:1500]
    # k_range_valid = range_all[1500:1900]

    #train_dataset_path = 'data/train_sim_data.mat'
    #valid_dataset_path = 'data/valid_sim_data.mat'

    k_range_train = range(0, 1500)
    k_range_valid = range(1500, 1900)
    
    train_dataset = sio.loadmat(train_dataset_path)
    valid_dataset = sio.loadmat(valid_dataset_path)

    #loss_fn = SO3FrobNorm()
    loss_fn = QuatLoss()

    #Float or Double?
    tensor_type = torch.float
    device = torch.device('cuda:0') if args.cuda else torch.device('cpu')

    model = QuaternionNet(D_in_sensor=2*train_dataset['y_k_j'].shape[0]*train_dataset['y_k_j'].shape[2])
    model.to(dtype=tensor_type, device=device)
    loss_fn.to(dtype=tensor_type, device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    #Load datasets
    normalization = compute_normalization(train_dataset).to(device)

    train_loader = DataLoader(A3DatasetQuatRel(train_dataset, k_range=k_range_train, normalization=normalization), 
                        batch_size=args.batch_size, pin_memory=True, 
                        shuffle=True, num_workers=1)
    valid_loader = DataLoader(A3DatasetQuatRel(valid_dataset, k_range=k_range_valid,normalization=normalization), 
                        batch_size=args.batch_size, pin_memory=True,
                        shuffle=False, num_workers=1)
    total_time = 0.
    now = datetime.datetime.now()
    start_datetime_str = '{}-{}-{}-{}-{}-{}'.format(now.year, now.month, now.day, now.hour, now.minute, now.second)


    #Configuration
    config = {
        'device': device
    }
    epoch_time = AverageMeter()
    avg_train_loss, train_ang_error, train_nll = validate(model, train_loader, loss_fn, config)
    avg_valid_loss, valid_ang_error, valid_nll = validate(model, valid_loader, loss_fn, config)
   
    for epoch in range(args.total_epochs):
        end = time.time()
        avg_train_loss = train(model, train_loader, loss_fn, optimizer, config)
        _, train_ang_error, train_nll = validate(model, train_loader, loss_fn, config)
        avg_valid_loss, valid_ang_error, valid_nll = validate(model, valid_loader, loss_fn, config)

        # Measure elapsed time
        epoch_time.update(time.time() - end)

        if epoch%args.epoch_display == 0:     
            print('Epoch {}. Loss (Train/Valid) {:.3f} / {:.3f} \t'
            'Train (Err/NLL) | Valid (Err/NLL) {:3.3f} / {:3.3f} | {:.3f} / {:3.3f}\t'
            'Epoch Time {epoch_time.val:.3f} (avg: {epoch_time.avg:.3f})'.format(
                epoch+1, avg_train_loss, avg_valid_loss, train_ang_error, train_nll, valid_ang_error, valid_nll, epoch_time=epoch_time))