import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.folder import default_loader
from liegroups.torch import SO3
import math
from utils import quaternion_from_matrix
import os
import os.path as osp
from PIL import Image
import pickle
import time
import cv2

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
            c_imgs = [osp.join(seq_dir, 'frame-{:06d}.color.png'.format(i)) for i in frame_idx]
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
            img = self.valid_jitter_transform(img)
            #img = torch.rand((1, 224, 224))
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


class KITTIVODataset(Dataset):
    """KITTI Odometry Benchmark dataset."""

    def __init__(self, kitti_data_pickle_file, transform_img=None, run_type='train'):
        """
        Args:
            kitti_data_pickle_file (string): Path to saved kitti dataset pickle.
            run_type (string): 'train', 'validate', or 'test'.
            transform_img (callable, optional): Optional transform to be applied to images.
        """
        self.pickle_file = kitti_data_pickle_file
        self.transform_img = transform_img
        self.load_kitti_data(run_type)  # Loads self.image_quad_paths and self.labels

    def load_kitti_data(self, run_type):
        with open(self.pickle_file, 'rb') as handle:
            kitti_data = pickle.load(handle)

        if run_type == 'train':

            self.image_quad_paths = kitti_data['train_img_paths_rgb']
            self.T_gt = kitti_data['train_T_gt']
            self.T_est = kitti_data['train_T_est']
            self.sequences = kitti_data['train_sequences']

        # elif run_type == 'validate' or run_type == 'valid':
        #     self.image_quad_paths = kitti_data['val_img_paths_rgb']
        #     self.T_gt = kitti_data['val_T_gt']
        #     self.T_est = kitti_data['val_T_est']
        #     self.sequence = kitti_data['val_sequence']
        #     self.tm_mat_path = kitti_data['val_tm_mat_path']

        elif run_type == 'test':
            self.image_quad_paths = kitti_data['test_img_paths_rgb']
            self.T_corr = kitti_data['test_T_corr']
            self.T_gt = kitti_data['test_T_gt']
            self.T_est = kitti_data['test_T_est']
            self.sequence = kitti_data['test_sequence']
            self.tm_mat_path = kitti_data['test_tm_mat_path']

        else:
            raise ValueError('run_type must be set to `train`, `validate` or `test`. ')

    def __len__(self):
        return len(self.image_quad_paths)

    def read_image(self, img_path):
        img = Image.open(img_path).convert('RGB')
        return img

    def __getitem__(self, idx):
        # Get all four images in the two pairs
        image_quad_paths = self.image_quad_paths[idx]
        #Note: transpose necessary so that targets are C_21 and not C_12
        target_quat = torch.from_numpy(quaternion_from_matrix(self.T_gt[idx].rot.as_matrix().transpose(0,1))).float()
        # Note: The camera y axis is facing down, hence 'yaw' of the vehicle, is 'pitch' of the camera
        if self.transform_img:
            image_pair = [self.transform_img(self.read_image(image_quad_paths[i])) for i in [0,2]]
        else:
            image_pair = [self.read_image(image_quad_paths[i]) for i in [0,2]]

        return image_pair, target_quat


class KITTIVODatasetPreTransformed(Dataset):
    """KITTI Odometry Benchmark dataset with full memory read-ins."""

    def __init__(self, kitti_dataset_file, seqs_base_path, transform_img=None, run_type='train'):
        self.kitti_dataset_file = kitti_dataset_file
        self.seqs_base_path = seqs_base_path

        self.transform_img = transform_img
        self.load_kitti_data(run_type)  # Loads self.image_quad_paths and self.labels

    def load_kitti_data(self, run_type):
        with open(self.kitti_dataset_file, 'rb') as handle:
            kitti_data = pickle.load(handle)

        if run_type == 'train':
            self.seqs = kitti_data['train_seqs']
            self.pose_indices = kitti_data['train_pose_indices']
            self.T_21_gt = kitti_data['train_T_21_gt']
            self.T_21_vo = kitti_data['train_T_21_vo']

        elif run_type == 'test':
            self.seqs = kitti_data['test_seqs']
            self.pose_indices = kitti_data['test_pose_indices']
            self.T_21_gt = kitti_data['test_T_21_gt']
            self.T_21_vo = kitti_data['test_T_21_vo']

        else:
            raise ValueError('run_type must be set to `train`, or `test`. ')

        print('Loading sequences...{}'.format(list(set(self.seqs))))
        self.seq_images = {seq: self.import_seq(seq) for seq in list(set(self.seqs))}
        print('...done loading images into memory.')

    def import_seq(self, seq):
        file_path = self.seqs_base_path + '/seq_{}.pt'.format(seq)
        data = torch.load(file_path)
        return data['im_l']

    def __len__(self):
        return len(self.T_21_gt)

    def prep_img(self, img):
        if self.transform_img is not None:
            return self.transform_img(img.float()/255.)
        else:
            return img.float() / 255.

    def compute_flow(self, img1, img2):
        #Convert back to W x H x C
        np_img1 = img1.permute(1,2,0).numpy()
        np_img2 = img2.permute(1,2,0).numpy()
        print(np_img1.shape)
        print(np_img2.shape)
        #print(np_img1)
        flow = cv2.calcOpticalFlowFarneback(np_img1, np_img2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        print(flow)
        return flow


    def __getitem__(self, idx):
        seq = self.seqs[idx]
        p_ids = self.pose_indices[idx]
        C_21_gt = self.T_21_gt[idx].rot.as_matrix()
        #C_21_err = self.T_21_gt[idx].rot.as_matrix().dot(self.T_21_vo[idx].rot.as_matrix().transpose())

        # image_pair = [self.prep_img(self.seq_images[seq][p_ids[0]]),
        #               self.prep_img(self.seq_images[seq][p_ids[1]])]
        flow_img = self.compute_flow(self.seq_images[seq][p_ids[0]], self.seq_images[seq][p_ids[1]])
        q_target = torch.from_numpy(quaternion_from_matrix(C_21_gt)).float()
        return flow_img, q_target