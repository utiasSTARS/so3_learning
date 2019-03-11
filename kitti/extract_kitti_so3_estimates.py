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
from loaders import KITTIVODataset
from torch.utils.data import Dataset, DataLoader
from vis import *
import torchvision.transforms as transforms


def run_so3_hydranet(trained_file_path, seq):
    # Float or Double?
    tensor_type = torch.float
    device = torch.device('cuda:1')
    loss_fn = QuatNLLLoss()
    batch_size = 32

    model = QuaternionDualCNN(num_hydra_heads=25)
    checkpoint = torch.load(trained_file_path)
    model.load_state_dict(checkpoint['full_model'])
    model.to(dtype=tensor_type, device=device)

    # Load datasets
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    kitti_data_pickle_file = 'datasets/monolith/kitti_data_sequence_{}.pickle'.format(seq)
    test_loader = DataLoader(KITTIVODataset(kitti_data_pickle_file, transform_img=transform, run_type='test'),
                              batch_size=batch_size, pin_memory=True,
                              shuffle=False, num_workers=8, drop_last=False)
    config = {
        'device': device
    }
    avg_valid_loss, valid_ang_error, valid_nll, predict_history = validate(model, test_loader, loss_fn, config, output_history=True)

    q_12 = predict_history[1]
    R_12 = SO3.from_quaternion(q_12).as_matrix()
    Sigma_12 = predict_history[2]

    torch.save({
        'R_12': R_12,
        'Sigma_12': Sigma_12,
    }, 'fusion/hydranet_output_model_seq_{}.pt'.format(seq))


if __name__ == '__main__':
    #Reproducibility
    #torch.manual_seed(7)
    #random.seed(72)
    seqs = ['00', '02', '05']
    trained_models_paths = ['best_model_seq_00_heads_25_epoch_9.pt',
                            'best_model_seq_02_heads_25_epoch_11.pt',
                            'best_model_seq_05_heads_25_epoch_12.pt'
                            ]
    base_path = './plots/'
    for model_path, seq in zip(trained_models_paths, seqs):
        run_so3_hydranet(base_path + model_path, seq)

