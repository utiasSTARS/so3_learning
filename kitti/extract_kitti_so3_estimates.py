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


def run_so3_hydranet(trained_file_path, seq):
    # Float or Double?
    tensor_type = torch.float
    device = torch.device('cuda:1')
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

    apply_blur = True
    kitti_data_pickle_file = 'datasets/obelisk/kitti_singlefile_data_sequence_{}.pickle'.format(seq)
    seqs_base_path = './'
    transform = None

    test_loader = DataLoader(KITTIVODatasetPreTransformed(kitti_data_pickle_file, seqs_base_path=seqs_base_path, transform_img=transform, run_type='test', apply_blur=apply_blur),
                              batch_size=batch_size, pin_memory=False,
                              shuffle=False, num_workers=4, drop_last=False)
    config = {
        'device': device
    }
    avg_valid_loss, valid_ang_error, valid_nll, predict_history = validate(model, test_loader, loss_fn, config, output_history=True)


    print('Extracted sequence {} \t' 
          '(Err/NLL) {:3.3f} / {:3.3f} \t'.format(
            seq, valid_ang_error, valid_nll))

    q_21 = predict_history[1]
    C_21 = SO3.from_quaternion(q_21).as_matrix()
    q_21_gt = predict_history[0]
    C_21_gt = SO3.from_quaternion(q_21_gt).as_matrix()

    Sigma_21 = predict_history[2]

    file_name = 'fusion/hydranet_output_model_seq_{}.pt'.format(seq)
    print('Outputting: {}'.format(file_name))
    torch.save({
        'Rot_21': C_21,
        'Rot_21_gt': C_21_gt,
        'Sigma_21': Sigma_21,
    }, file_name)


if __name__ == '__main__':
    #Reproducibility
    #torch.manual_seed(7)
    #random.seed(72)
    seqs = ['00', '02', '05']
    trained_models_paths = ['best_model_seq_00_heads_25_epoch_10.pt',
                            'best_model_seq_02_heads_25_epoch_27.pt',
                            'best_model_seq_05_heads_25_epoch_23.pt'
                            ]
    base_path = './plots/'
    for model_path, seq in zip(trained_models_paths, seqs):
        run_so3_hydranet(base_path + model_path, seq)

