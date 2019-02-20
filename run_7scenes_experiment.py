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
import time, sys
import argparse
import datetime
from train_test import *
from loaders import SevenScenesData
from torch.utils.data import Dataset, DataLoader
from vis import *
import torchvision.transforms as transforms

if __name__ == '__main__':
    #Reproducibility
    #torch.manual_seed(7)
    #random.seed(72)

    parser = argparse.ArgumentParser(description='3D training arguments.')
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epoch_display', type=int, default=1)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--total_epochs', type=int, default=100)
    parser.add_argument('--num_heads', type=int, default=1)
    parser.add_argument('--scene', type=str, default='chess')

    args = parser.parse_args()


    #loss_fn = SO3FrobNorm()
    #loss_fn = QuatLoss()
    loss_fn = QuatNLLLoss()
    #loss_fn = SO3NLLLoss()

    #Float or Double?
    tensor_type = torch.float
    device = torch.device('cuda:0') if args.cuda else torch.device('cpu')


    num_hydra_heads=args.num_heads
    model = QuaternionCNN(num_hydra_heads=num_hydra_heads)
    #model.sensor_net.freeze_layers()

    model.to(dtype=tensor_type, device=device)



    loss_fn.to(dtype=tensor_type, device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    #Load datasets
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    transform_jitter = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.45, hue=0.45),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_loader = DataLoader(SevenScenesData(args.scene, '/home/valentinp/research/data/7scenes', train=True, transform=transform),
                        batch_size=args.batch_size, pin_memory=True,
                        shuffle=True, num_workers=10, drop_last=False)
    valid_loader = DataLoader(SevenScenesData(args.scene, '/home/valentinp/research/data/7scenes', train=False, transform=transform, valid_jitter_transform=transform_jitter),
                        batch_size=args.batch_size, pin_memory=True,
                        shuffle=False, num_workers=10, drop_last=False)
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
    plot_errors_with_sigmas(predict_history[0], predict_history[1], predict_history[2], filename=sigma_filename)

    print('Starting Training \t' 
          'Train (Err/NLL) | Valid (Err/NLL) {:3.3f} / {:3.3f} | {:.3f} / {:3.3f}\t'.format(
            train_ang_error, train_nll, valid_ang_error, valid_nll))

    best_valid_err = valid_ang_error
    for epoch in range(args.total_epochs):
        end = time.time()
        avg_train_loss = train(model, train_loader, loss_fn, optimizer, config)

        _, train_ang_error, train_nll = validate(model, train_loader, loss_fn, config)
        avg_valid_loss, valid_ang_error, valid_nll, predict_history = validate(model, valid_loader, loss_fn, config, output_history=True)

        # Measure elapsed time
        epoch_time.update(time.time() - end)

        # if epoch == 5:
        #     print('Freezing the ResNet!')
        #     model.sensor_net.freeze_layers()

        if valid_ang_error < best_valid_err:
            print('New best validation angular error! Outputting plots and saving model.')
            torch.save({
                'full_model': model.state_dict(),
                'sensor_net': model.sensor_net.state_dict(),
                'direct_covar_head': model.direct_covar_head.state_dict(),
                'epoch': epoch+1,
            }, '7scenes/best_model_{}_heads_{}_epoch_{}.pt'.format(args.scene,model.num_hydra_heads, epoch+1))
            best_valid_err = valid_ang_error
            sigma_filename = '7scenes/sigma_plot_{}_heads_{}_epoch_{}.pdf'.format(args.scene, model.num_hydra_heads, epoch+1)
            nees_filename = '7scenes/nees_plot_{}_heads_{}_epoch_{}.pdf'.format(args.scene, model.num_hydra_heads, epoch+1)

            plot_errors_with_sigmas(predict_history[0], predict_history[1], predict_history[2], filename=sigma_filename)
            plot_nees(predict_history[0], predict_history[1], predict_history[2], filename=nees_filename)


        if epoch%args.epoch_display == 0:
            print('Epoch {}. Loss (Train/Valid) {:.3E} / {:.3E} \t'
            'Train (Err/NLL) | Valid (Err/NLL) {:3.3f} / {:3.3f} | {:.3f} / {:3.3f}\t'
            'Epoch Time {epoch_time.val:.3f} (avg: {epoch_time.avg:.3f})'.format(
                epoch+1, avg_train_loss, avg_valid_loss, train_ang_error, train_nll, valid_ang_error, valid_nll, epoch_time=epoch_time))

