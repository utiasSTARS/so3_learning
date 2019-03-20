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
from utils import *
import glob
import csv

if __name__ == '__main__':
#    ref = {'chess': '13',
#           'fire': '4',
#           'heads': '11',
#           'office':  '15'  ,
#           'pumpkin':   ,
#           'kitchen':   ,
#           'stairs':    ,
#           }
    #Reproducibility
    #torch.manual_seed(7)
    #random.seed(72)
    experiment = 'resnet34'
    scenes = ['chess', 'fire', 'heads', 'office', 'pumpkin', 'redkitchen', 'stairs']
#    scenes = ['stairs']
    csv_header = ['Scene', 'epoch', 'NLL', 'ang_error']
    stats_list = []
    for scene in scenes:
        print(scene)
        best_nll = 1e5
        for scene_checkpoint in glob.glob('7scenes/'+experiment + '/' + scene + '/**.pt'):
#            print(scene_checkpoint)
            check_point = torch.load(scene_checkpoint, map_location=lambda storage, loc: storage) #The lambda call esnures
            (q_gt, q_est, R_est, R_direct_est) = (check_point['predict_history'][0],
                                                  check_point['predict_history'][1],
                                                  check_point['predict_history'][2],
                                                  check_point['predict_history'][3])
        
            nll = nll_quat(q_est, q_gt, R_est.inverse()).mean()
            angular_error = quat_ang_error(q_est, q_gt).mean()*(180./3.1415)
#            print(angular_error, nll)
            if nll < best_nll:
                best_error = float(angular_error)
                best_epoch = check_point['epoch']
                best_nll = float(nll)

        stats_list.append([scene, best_epoch, best_nll, best_error])
        
        
        
    csv_filename = '7scenes/' + experiment + '/stats.csv'
    with open(csv_filename, "w") as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)
        writer.writerows(stats_list)
        
    
#    fig, ax = plt.subplots(3, 1, sharex='col', sharey='row', figsize=(6, 8))
#
#    x_labels =np.arange(0, q_gt.shape[0])
#    phi_errs = quat_log_diff(q_est, q_gt).numpy()
#    R_est = R_est.numpy()
#    R_direct_est = R_direct_est.numpy()
#    font_size = 18
#
#
#    _plot_sigma(x_labels, phi_errs[:, 0], 0., np.sqrt(R_est[:, 0, 0].flatten()),
#                np.sqrt(R_direct_est[:, 0, 0].flatten()), '$\phi_1$ err', ax[0], font_size=font_size)
#    _plot_sigma(x_labels, phi_errs[:, 1], 0., np.sqrt(R_est[:, 1, 1].flatten()),
#                np.sqrt(R_direct_est[:, 1, 1].flatten()), '$\phi_2$ err', ax[1], font_size=font_size)
#    _plot_sigma(x_labels, phi_errs[:, 2], 0., np.sqrt(R_est[:, 2, 2].flatten()),
#                np.sqrt(R_direct_est[:, 2, 2].flatten()), '$\phi_3$ err', ax[2], font_size=font_size)
    ##ax[2].legend(fontsize=font_size, loc='center')
    ##image_array = canvas_to_array(fig)
#    ax[2].xaxis.set_tick_params(labelsize=font_size-2)
#    ax[0].yaxis.set_tick_params(labelsize=font_size-2)
#    ax[1].yaxis.set_tick_params(labelsize=font_size-2)
#    ax[2].yaxis.set_tick_params(labelsize=font_size-2)
#    ax[2].set_xlabel('Pose', fontsize=font_size)
#
#    fig_name = scene_checkpoint.split('/')[1].split('.')[0] + '.png'
#    fig.savefig(fig_name, bbox_inches='tight', dpi=300)