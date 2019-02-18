import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from liegroups.torch import SO3

global KITTI_SEQS_DICT
KITTI_SEQS_DICT = {'00': {'date': '2011_10_03',
            'drive': '0027',
            'frames': range(0, 4541)},
        '01': {'date': '2011_10_03',
            'drive': '0042',
            'frames': range(0, 1101)},
        '02': {'date': '2011_10_03',
            'drive': '0034',
            'frames': range(0, 4661)},
        '04': {'date': '2011_09_30',
            'drive': '0016',
            'frames': range(0, 271)},
        '05': {'date': '2011_09_30',
            'drive': '0018',
            'frames': range(0, 2761)},
        '06': {'date': '2011_09_30',
            'drive': '0020',
            'frames': range(0, 1101)},
        '07': {'date': '2011_09_30',
            'drive': '0027',
            'frames': range(0, 1101)},
        '08': {'date': '2011_09_30',
            'drive': '0028',
            'frames': range(1100, 5171)},
        '09': {'date': '2011_09_30',
            'drive': '0033',
            'frames': range(0, 1591)},
        '10': {'date': '2011_09_30',
            'drive': '0034',
            'frames': range(0, 1201)}}

class KITTIOdometryDataset(Dataset):
    """KITTI Odometry Benchmark dataset."""

    def __init__(self, kitti_data_pickle_file, img_type='rgb', transform_img=None, run_type='train', target_type='corr'):
        """
        Args:
            kitti_data_pickle_file (string): Path to saved kitti dataset pickle.
            run_type (string): 'train', 'validate', or 'test'.
            transform_img (callable, optional): Optional transform to be applied to images.
        """
        self.pickle_file = kitti_data_pickle_file
        self.transform_img = transform_img
        self.img_type = img_type
        self.load_kitti_data(run_type) #Loads self.image_quad_paths and self.labels
        self.target_type = target_type #corr or gt

    def load_kitti_data(self, run_type):
        with open(self.pickle_file, 'rb') as handle:
            kitti_data = pickle.load(handle)

        #Empirical precision matrix (inverse covariance) computed over the training data
        self.train_se3_precision = torch.from_numpy(kitti_data.train_se3_precision).float()
        self.train_pose_deltas = kitti_data.train_pose_deltas
        self.test_pose_delta = kitti_data.test_pose_delta

        if run_type == 'train':

            self.image_quad_paths = kitti_data.train_img_paths_rgb if self.img_type=='rgb' else kitti_data.train_img_paths_mono
            self.T_corr = kitti_data.train_T_corr
            self.T_gt = kitti_data.train_T_gt
            self.T_est = kitti_data.train_T_est
            self.sequences = kitti_data.train_sequences
            
        elif run_type == 'validate' or run_type == 'valid':
            self.image_quad_paths = kitti_data.val_img_paths_rgb if self.img_type=='rgb' else kitti_data.val_img_paths_mono
            self.T_corr = kitti_data.val_T_corr
            self.T_gt = kitti_data.val_T_gt
            self.T_est = kitti_data.val_T_est
            self.sequence = kitti_data.val_sequence
            self.tm_mat_path = kitti_data.val_tm_mat_path

        elif run_type == 'test':    
            self.image_quad_paths = kitti_data.test_img_paths_rgb if self.img_type=='rgb' else kitti_data.test_img_paths_mono
            self.T_corr = kitti_data.test_T_corr
            self.T_gt = kitti_data.test_T_gt
            self.T_est = kitti_data.test_T_est
            self.sequence = kitti_data.test_sequence
            self.tm_mat_path = kitti_data.test_tm_mat_path

        else:
            raise ValueError('run_type must be set to `train`, `validate` or `test`. ')


    def __len__(self):
        return len(self.image_quad_paths)

    def read_image(self, img_path):
        img = Image.open(img_path).convert('RGB')
        return img

    def __getitem__(self, idx):
        #Get all four images in the two pairs
        image_quad_paths = self.image_quad_paths[idx]
        if self.target_type == 'corr':
            T_target = self.T_corr
        elif self.target_type == 'est':
            T_target = self.T_est
        else:
            T_target = self.T_gt

        target_se3 = torch.from_numpy(T_target[idx].as_matrix()).float()
        target_rot = torch.from_numpy(T_target[idx].rot.as_matrix()).float()
        #Note: The camera y axis is facing down, hence 'yaw' of the vehicle, is 'pitch' of the camera
        target_yaw = torch.Tensor([self.T_gt[idx].rot.to_rpy()[1] - self.T_est[idx].rot.to_rpy()[1]]).float()

        if self.transform_img:
            image_quad = [self.transform_img(self.read_image(image_quad_paths[i])) for i in range(4)]
        else:
            image_quad = [self.read_image(image_quad_paths[i]) for i in range(4)]

        return image_quad, target_rot, target_yaw, target_se3

    
def quaternion_from_matrix(matrix, isprecise=False):
    """Return quaternion from rotation matrix.

    If isprecise is True, the input matrix is assumed to be a precise rotation
    matrix and a faster algorithm is used.
    """

    M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
    if isprecise:
        q = np.empty((4, ))
        t = np.trace(M)
        if t > M[3, 3]:
            q[0] = t
            q[3] = M[1, 0] - M[0, 1]
            q[2] = M[0, 2] - M[2, 0]
            q[1] = M[2, 1] - M[1, 2]
        else:
            i, j, k = 0, 1, 2
            if M[1, 1] > M[0, 0]:
                i, j, k = 1, 2, 0
            if M[2, 2] > M[i, i]:
                i, j, k = 2, 0, 1
            t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
            q[i] = t
            q[j] = M[i, j] + M[j, i]
            q[k] = M[k, i] + M[i, k]
            q[3] = M[k, j] - M[j, k]
            q = q[[3, 0, 1, 2]]
        q *= 0.5 / math.sqrt(t * M[3, 3])
    else:
        m00 = M[0, 0]
        m01 = M[0, 1]
        m02 = M[0, 2]
        m10 = M[1, 0]
        m11 = M[1, 1]
        m12 = M[1, 2]
        m20 = M[2, 0]
        m21 = M[2, 1]
        m22 = M[2, 2]
        # symmetric matrix K
        K = np.array([[m00-m11-m22, 0.0,         0.0,         0.0],
                         [m01+m10,     m11-m00-m22, 0.0,         0.0],
                         [m02+m20,     m12+m21,     m22-m00-m11, 0.0],
                         [m21-m12,     m02-m20,     m10-m01,     m00+m11+m22]])
        K /= 3.0
        # quaternion is eigenvector of K that corresponds to largest eigenvalue
        w, V = np.linalg.eigh(K)
        q = V[[3, 0, 1, 2], np.argmax(w)]
    if q[0] < 0.0:
        np.negative(q, q)
    return q

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