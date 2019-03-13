import numpy as np
import torch

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
from loaders import KITTIVODataset, KITTIVODatasetPreTransformed
from torch.utils.data import Dataset, DataLoader
from vis import *
import torchvision.transforms as transforms

if __name__ == '__main__':
    #Reproducibility
    #torch.manual_seed(7)
    #random.seed(72)

    parser = argparse.ArgumentParser(description='3D training arguments.')
    parser.add_argument('--seq', type=str, default='00')
    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epoch_display', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--total_epochs', type=int, default=15)
    parser.add_argument('--num_heads', type=int, default=25)
    parser.add_argument('--q_target_sigma', type=float, default=0.)
    parser.add_argument('--freeze_body', action='store_true', default=False)

    args = parser.parse_args()
    print(args)


    #loss_fn = SO3FrobNorm()
    #loss_fn = QuatLoss()
    loss_fn = QuatNLLLoss()
    #loss_fn = SO3NLLLoss()

    #Float or Double?
    tensor_type = torch.float
    device = torch.device('cuda:1') if args.cuda else torch.device('cpu')


    num_hydra_heads=args.num_heads
    model = QuaternionDualCNN(num_hydra_heads=num_hydra_heads)
    if args.freeze_body:
        model.sensor_net.freeze_layers()

    model.to(dtype=tensor_type, device=device)
    loss_fn.to(dtype=tensor_type, device=device)


    optimizer = torch.optim.Adam(
        [{'params': model.sensor_net.parameters(), 'lr': 0.1*args.lr},
        #{'params': model.sensor_net1.parameters(), 'lr': 0.1*args.lr},
        {'params': model.heads.parameters()},
         {'params': model.direct_covar_head.parameters()}],
        lr=args.lr)

    #Load datasets
    # transform = transforms.Compose([
    #     transforms.Resize(224),
    #     transforms.CenterCrop(224),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                          std=[0.229, 0.224, 0.225])
    # ])



    # kitti_data_pickle_file = 'kitti/datasets/obelisk/kitti_data_sequence_{}.pickle'.format(args.seq)

    # train_loader = DataLoader(KITTIVODataset(kitti_data_pickle_file, transform_img=transform, run_type='train'),
    #                     batch_size=args.batch_size, pin_memory=True,
    #                     shuffle=True, num_workers=12, drop_last=True)
    # valid_loader = DataLoader(KITTIVODataset(kitti_data_pickle_file, transform_img=transform, run_type='test'),
    #                           batch_size=args.batch_size, pin_memory=True,
    #                           shuffle=False, num_workers=12, drop_last=False)

    transform = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    kitti_data_pickle_file = 'kitti/datasets/obelisk/kitti_singlefile_data_sequence_{}.pickle'.format(args.seq)

    seqs_base_path = 'kitti'
    train_loader = DataLoader(KITTIVODatasetPreTransformed(kitti_data_pickle_file, seqs_base_path=seqs_base_path, transform_img=transform, run_type='train'),
                              batch_size=args.batch_size, pin_memory=False,
                              shuffle=True, num_workers=8, drop_last=True)

    valid_loader = DataLoader(KITTIVODatasetPreTransformed(kitti_data_pickle_file, seqs_base_path=seqs_base_path, transform_img=transform, run_type='test'),
                              batch_size=args.batch_size, pin_memory=False,
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
    avg_valid_loss, valid_ang_error, valid_nll, predict_history = validate(model, valid_loader, loss_fn, config, output_history=True, output_grid=True)

    #Visualize
    sigma_filename = 'kitti/plots/sigma_plot_heads_{}_epoch_{}.pdf'.format(model.num_hydra_heads, 0)
    plot_errors_with_sigmas(predict_history[0], predict_history[1], predict_history[2], predict_history[3], filename=sigma_filename)

    print('Starting Training \t' 
          'Train (Err/NLL) | Valid (Err/NLL) {:3.3f} / {:3.3f} | {:.3f} / {:3.3f}\t'.format(
            train_ang_error, train_nll, valid_ang_error, valid_nll))

    best_valid_loss = avg_valid_loss
    for epoch in range(args.total_epochs):
        end = time.time()
        avg_train_loss = train(model, train_loader, loss_fn, optimizer, config, q_target_sigma=args.q_target_sigma)

        _, train_ang_error, train_nll, predict_history_train = validate(model, train_loader, loss_fn, config, output_history=True)
        avg_valid_loss, valid_ang_error, valid_nll, predict_history = validate(model, valid_loader, loss_fn, config, output_history=True)

        # Measure elapsed time
        epoch_time.update(time.time() - end)

        # if epoch == 5:
        #     print('Freezing the ResNet!')
        #     model.sensor_net.freeze_layers()

        if avg_valid_loss < best_valid_loss:
            print('New best validation loss! Outputting plots and saving model.')

            best_valid_loss = avg_valid_loss

            sigma_filename = 'kitti/plots/error_sigma_plot_seq_{}_heads_{}_epoch_{}.pdf'.format(args.seq, model.num_hydra_heads, epoch+1)
            #nees_filename = 'kitti/plots/nees_plot_heads_{}_epoch_{}.pdf'.format(model.num_hydra_heads, epoch+1)

            plot_errors_with_sigmas(predict_history[0], predict_history[1], predict_history[2], predict_history[3], filename=sigma_filename)

            abs_filename = 'kitti/plots/abs_sigma_plot_seq_{}_heads_{}_epoch_{}.pdf'.format(args.seq, model.num_hydra_heads,
                                                                                  epoch + 1)
            plot_abs_with_sigmas(predict_history[0], predict_history[1], predict_history[2], predict_history[3],
                                    filename=abs_filename)

            torch.save({
                'full_model': model.state_dict(),
                'predict_history': predict_history,
                'epoch': epoch + 1,
            }, 'kitti/plots/best_model_seq_{}_heads_{}_epoch_{}.pt'.format(args.seq, model.num_hydra_heads, epoch + 1))


            #plot_nees(predict_history[0], predict_history[1], predict_history[2], filename=nees_filename)


        if epoch%args.epoch_display == 0:
            print('Epoch {}. Loss (Train/Valid) {:.3E} / {:.3E} \t'
            'Train (Err/NLL) | Valid (Err/NLL) {:3.3f} / {:3.3f} | {:.3f} / {:3.3f}\t'
            'Epoch Time {epoch_time.val:.3f} (avg: {epoch_time.avg:.3f})'.format(
                epoch+1, avg_train_loss, avg_valid_loss, train_ang_error, train_nll, valid_ang_error, valid_nll, epoch_time=epoch_time))

