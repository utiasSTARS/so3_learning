import numpy as np
import torch
from torch import autograd
import os


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.io as sio
import math
import sys
sys.path.insert(0,'..')
from models import *
from loss import *
import time, sys
import argparse
import datetime
from train_test import *
from loaders import KITTIVODatasetPreTransformed
from torch.utils.data import Dataset, DataLoader
from vis import *
import torchvision.transforms as transforms


def run_so3_hydranet(trained_file_path, seq, kitti_data_file=None):
    # Float or Double?
    tensor_type = torch.float
    device = torch.device('cuda:0')
    loss_fn = QuatNLLLoss()
    batch_size = 32

    model = QuaternionCNN(num_hydra_heads=25)
    checkpoint = torch.load(trained_file_path)
    model.load_state_dict(checkpoint['full_model'])
    model.to(dtype=tensor_type, device=device)

    # Load datasets
    # transform = transforms.Compose([
    #     transforms.Resize(224),
    #     transforms.CenterCrop(224),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                          std=[0.229, 0.224, 0.225])
    # ])

    apply_blur = False
    seqs_base_path = './data'
    seq_prefix = 'seq_noncropped_'
    kitti_data_pickle_file = kitti_data_file
    transform = None

    test_loader = DataLoader(KITTIVODatasetPreTransformed(kitti_data_pickle_file, seqs_base_path=seqs_base_path, transform_img=transform,
                                                          run_type='test', apply_blur=apply_blur, seq_prefix=seq_prefix, use_only_seq=seq),
                              batch_size=batch_size, pin_memory=False,
                              shuffle=False, num_workers=4, drop_last=False)

    test_loader_reverse = DataLoader(KITTIVODatasetPreTransformed(kitti_data_pickle_file, seqs_base_path=seqs_base_path, transform_img=transform,
                                                                  run_type='test', apply_blur=apply_blur,seq_prefix=seq_prefix, reverse_images=True, use_only_seq=seq),
                              batch_size=batch_size, pin_memory=False,
                              shuffle=False, num_workers=0, drop_last=False)

    config = {
        'device': device
    }
    avg_valid_loss, valid_ang_error, valid_nll, predict_history = validate(model, test_loader, loss_fn, config, output_history=True)


    print('Extracted sequence {} \t'
          '(Err/NLL) {:3.3f} / {:3.3f} \t'.format(
            seq, valid_ang_error, valid_nll))



    avg_valid_loss, valid_ang_error, valid_nll, predict_history_reverse = validate(model, test_loader_reverse, loss_fn,
                                                                                   config, output_history=True)
    print('Extracted reverse sequence {} \t'
          '(Err/NLL) {:3.3f} / {:3.3f} \t'.format(
        seq, valid_ang_error, valid_nll))

    q_21 = predict_history[1]
    C_21 = SO3.from_quaternion(q_21).as_matrix()

    q_12 = predict_history_reverse[1]
    C_12 = SO3.from_quaternion(q_12).as_matrix()


    q_21_gt = predict_history[0]
    C_21_gt = SO3.from_quaternion(q_21_gt).as_matrix()

    Sigma_21 = predict_history[2]
    Sigma_12 = predict_history_reverse[2]

    file_name = 'fusion/hydranet_output_reverse_model_seq_{}.pt'.format(seq)
    print('Outputting: {}'.format(file_name))
    torch.save({
        'Rot_21': C_21,
        'Sigma_21': Sigma_21,
        'Rot_12': C_12,
        'Sigma_12': Sigma_12,
        'Rot_21_gt': C_21_gt,
    }, file_name)


if __name__ == '__main__':
    #Reproducibility
    #torch.manual_seed(7)
    #random.seed(72)
    seqs = ['02']
    trained_models_paths = ['best_model_seq_02_delta_1_heads_25_epoch_10.pt']
    kitti_data_file = None
    base_path = './plots_and_models/flow_large/'
    # seqs = ['09', '10']
    # kitti_data_file = 'datasets/obelisk/kitti_singlefile_data_sequence_0910_delta_1_reverse_True.pickle'
    # trained_models_paths = ['best_model_seq_0910_delta_1_heads_25_epoch_18.pt',
    #                         'best_model_seq_0910_delta_1_heads_25_epoch_18.pt']

    for model_path, seq in zip(trained_models_paths, seqs):
        kitti_data_file = 'datasets/obelisk/kitti_singlefile_data_sequence_{}_delta_1_reverse_True_minta_0.03.pickle'.format(seq)
        run_so3_hydranet(base_path + model_path, seq, kitti_data_file=kitti_data_file)

