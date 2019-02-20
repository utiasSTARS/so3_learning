import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.folder import default_loader
from liegroups.torch import SO3
import math
from utils import quaternion_from_matrix
import os
import os.path as osp

class PlanetariumData(Dataset):
    """Synthetic data"""

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

        if (torch.isnan(self.q_target).any()):
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

class SevenScenesData(Dataset):
    def __init__(self, scene, data_path, train, transform=None, valid_jitter_transform=None):
        
        """
          :param scene: scene name: 'chess', 'pumpkin', ...
          :param data_path: root 7scenes data directory.

        """
        self.transform = transform
        self.valid_jitter_transform = valid_jitter_transform
        self.train = train
          # directories
        base_dir = osp.join(osp.expanduser(data_path), scene)   
          # decide which sequences to use
        if train:
            split_file = osp.join(base_dir, 'TrainSplit.txt')
        else:
            split_file = osp.join(base_dir, 'TestSplit.txt')
        with open(split_file, 'r') as f:
            seqs = [int(l.split('sequence')[-1]) for l in f if not l.startswith('#')]
    
          # read poses and collect image names
        self.c_imgs = []
        self.d_imgs = []
        self.pose_files = []
        self.gt_idx = np.empty((0,), dtype=np.int)
        ps = {}
        for seq in seqs:
            seq_dir = osp.join(base_dir, 'seq-{:02d}'.format(seq))
            p_filenames = [n for n in os.listdir(osp.join(seq_dir, '.')) if n.find('pose') >= 0]
            frame_idx = np.array(range(len(p_filenames)), dtype=np.int)
            pss = [np.loadtxt(osp.join(seq_dir, 'frame-{:06d}.pose.txt'.format(i))).flatten() for i in frame_idx]
            ps[seq] = np.asarray(pss)
            c_imgs = [osp.join(seq_dir, 'frame-{:06d}.depth.png'.format(i)) for i in frame_idx]
            self.c_imgs.extend(c_imgs)
        self.poses = np.empty((0,16))
        for seq in seqs:
            self.poses = np.vstack((self.poses,ps[seq]))

        print('Loaded {} poses'.format(self.poses.shape[0]))

    def __getitem__(self, index):
        img = self.load_image(self.c_imgs[index])
        pose = self.poses[index].reshape((4,4))
        rot = pose[0:3,0:3] #Poses are camera to world, we need world to camera


        if (not self.train) and (self.valid_jitter_transform is not None) and index > self.poses.shape[0] / 2:
            #img = self.valid_jitter_transform(img)
            img = torch.rand((1, 224, 224))
        else:
            if self.transform:
                img = self.transform(img)



        return img, torch.from_numpy(quaternion_from_matrix(rot.T)).float()

    def __len__(self):
        return self.poses.shape[0]

    def load_image(self, filename, loader=default_loader):
        try:
            img = loader(filename)
        except IOError as e:
            print('Could not load image {:s}, IOError: {:s}'.format(filename, e))
            return None
        except:
            print('Could not load image {:s}, unexpected error'.format(filename))
            return None
        return img