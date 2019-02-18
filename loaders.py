import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from liegroups.torch import SO3
import math
from utils import quaternion_from_matrix

class PlanetariumData(Dataset):
    """Assignment 3 Dataset."""

    def __init__(self, dataset, k_range, normalization=1., mat_targets=False):
        self.dataset = dataset
        self.norm = normalization
        self.load_data(k_range)
        self.mat_targets = mat_targets

    def load_data(self, k_range):

        C_gt = self.dataset['T_vk_i'][k_range, :3, :3]
        q_target = np.empty((C_gt.shape[0], 4))
        for i in range(C_gt.shape[0]):
            q_target[i] = quaternion_from_matrix(C_gt[i])

        self.q_target = torch.from_numpy(q_target).float()
        self.C_target = torch.from_numpy(C_gt).float()
        #C_gt = torch.from_numpy(self.dataset['T_vk_i'][k_range, :3, :3]).float()
        #C_gt = SO3.from_matrix(C_gt, normalize=True)

        #self.q_target = C_gt.to_quaternion()
        if (torch.isnan(self.q_target).any()):
            # print(torch.isnan(self.q_target[:,0]).nonzero())
            # print(C_gt.as_matrix()[torch.isnan(self.q_target[:,0])])
            raise Exception('Quaternions have nan at indices: {}'.format(torch.isnan(self.q_target[:,0]).nonzero()))

        y =  torch.from_numpy(self.dataset['y_k_j'][:, k_range, :]).float()
        self.sensor_data = y

    def __len__(self):
        return len(self.q_target)

    def __getitem__(self, idx):
        y = self.sensor_data[:, idx, :].clone()
        if self.norm is not None:
            y[:, y[0, :] > 0] = y[:, y[0, :] > 0]/self.norm

        if self.mat_targets:
            target = self.C_target[idx].clone()
        else:
            target = self.q_target[idx].clone()
        return y.transpose(0,1).flatten(), target