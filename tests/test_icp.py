import torch
import numpy as np
import scipy.io as sio
from liegroups.torch import SE3, SO3

#dataset_path = '/Users/valentinp/research/data/lkf/kitti/kitti-seq-00-with-icp-noiseless.mat'
dataset_path = 'data/train_sim_data.mat'
k_range = range(0, 1900)
dataset = sio.loadmat(dataset_path)


#Load data
y_motion = torch.from_numpy(np.vstack((-dataset['v_vk_vk_i'], -dataset['w_vk_vk_i'])))
t = dataset['t'].flatten()
    
#Initialize state
T = SE3.from_matrix(torch.from_numpy(dataset['T_vk_i'][k_range[0]]))

#History 
k_num = len(list(k_range))
T_hist = torch.empty((k_num, 4, 4))
T_hist[0] = T.as_matrix()

for (rng_i, k) in enumerate(k_range[1:]):
    #We are starting with the second state in our range
    k_i = rng_i + 1

    #Prediction
    dt = t[k_range[k_i]] - t[k_range[k_i-1]]
    
    #Propagate nominal state
    xi = dt*y_motion[:, k-1]
    T = SE3.exp(xi).dot(T)    

    #History
    T_hist[k_i] = T.as_matrix()

C_hist_1 = T_hist[1:, :3, :3]
C_hist_0 = T_hist[:-1, :3, :3]
C_hist = C_hist_0.bmm(C_hist_1.transpose(1,2))

C_gt_1 = torch.from_numpy(dataset['T_vk_i'][k_range[1:], :3, :3]).float()
C_gt_0 = torch.from_numpy(dataset['T_vk_i'][k_range[:-1], :3, :3]).float()
C_gt = C_gt_0.bmm(C_gt_1.transpose(1,2))


C_err = C_hist.bmm(C_gt.transpose(1,2))
angular_error = SO3.from_matrix(C_err).log().norm(dim=1).mean()*(180./3.1415)
print(angular_error)