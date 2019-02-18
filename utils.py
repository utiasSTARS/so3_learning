import torch
import numpy as np
import matplotlib.pyplot as plt
import math
from lie_algebra_full import so3_wedge, so3_log

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def compute_normalization(dataset):
    
    y_obs = torch.from_numpy(dataset['y_k_j']).float()

    obs_dim = y_obs.shape[0]
    obs_mean = y_obs.new_empty((obs_dim,1))

    for d in range(obs_dim):
        obs_mean[d] = y_obs[d, y_obs[d, :,:] > -1].abs().mean()

    return obs_mean

def compute_error_angles(C_est, C_gt):
    frob_norms = (C_est - C_gt).norm(dim=(1,2)).clamp(0,2.*math.sqrt(2))
    error_angles = 2*torch.asin(frob_norms/(2.*math.sqrt(2)))
    return error_angles


def isclose(mat1, mat2, tol=1e-6):
    """Check element-wise if two tensors are close within some tolerance.

    Either tensor can be replaced by a scalar.
    """
    return (mat1 - mat2).abs_().lt(tol)

#NxD -> NxD, normalize N vectorss
def normalize_vecs(vecs):
    if vecs.dim() < 2:
        vecs = vecs.unsqueeze(0)
    normed_vecs = vecs/vecs.norm(dim=1,keepdim=True)
    return normed_vecs.squeeze_()

#NxMxD -> NxDxD 
#Computes sample covariances (assuming zero mean) over each of N batches with M D-dimensional vectors
def batch_sample_covariance(vecs):
    if vecs.dim() < 3:
        vecs = vecs.unsqueeze(0)

    M = vecs.shape[1] #Assumes constant amount of vectors
    covars = (1./(M - 1))*vecs.transpose(1,2).bmm(vecs)
    return covars

#Quaternion difference of two unit quaternions
def quat_norm_diff(q_a, q_b):
    if q_a.dim() < 2:
        q_a = q_a.unsqueeze(0)
    if q_b.dim() < 2:
        q_b = q_b.unsqueeze(0)
    return torch.min((q_a-q_b).norm(dim=1), (q_a+q_b).norm(dim=1)).squeeze_()


def quat_log(q):
    #input: q: Nx4
    #output: Log(q) Nx3 (see Sola eq. 105a/b)

    if q.dim() < 2:
        q = q.unsqueeze(0)

    #Check for negative scalars first, then substitute q for -q whenever that is the case (this accounts for the double cover of S3 over SO(3))
    neg_angle_mask = q[:, 0] < 0.
    neg_angle_inds = neg_angle_mask.nonzero().squeeze_(dim=1)

    q_w = q[:, 0].clone()
    q_v = q[:, 1:].clone()

    if len(neg_angle_inds) > 0:
        q_w[neg_angle_inds] = -1.*q_w[neg_angle_inds]
        q_v[neg_angle_inds] = -1.*q_v[neg_angle_inds]

    q_v_norm = q_v.norm(dim=1)

    # Near phi==0 (q_w ~ 1), use first order Taylor expansion
    angles = 2. * torch.atan2(q_v_norm, q_w) 
    small_angle_mask = isclose(angles, 0.)
    small_angle_inds = small_angle_mask.nonzero().squeeze_(dim=1)

    phi = q.new_empty((q.shape[0], 3))

    

    if len(small_angle_inds) > 0:
        q_v_small = q_v[small_angle_inds]
        q_v_n_small = q_v_norm[small_angle_inds].unsqueeze(1)
        q_w_small = q_w[small_angle_inds].unsqueeze(1)   
        phi[small_angle_inds, :] = \
            2. * ( q_v_small /  q_w_small) * \
            (1 - ( q_v_n_small ** 2)/(3. * ( q_w_small ** 2)))
            

    # Otherwise...
    large_angle_mask = 1 - small_angle_mask  # element-wise not
    large_angle_inds = large_angle_mask.nonzero().squeeze_(dim=1)
    
    if len(large_angle_inds) > 0:
        angles_large = angles[large_angle_inds]
        #print(q_v[large_angle_inds].shape)
        #print(q_v_norm[large_angle_inds].shape)
        
        axes = q_v[large_angle_inds] / q_v_norm[large_angle_inds].unsqueeze(1)
        phi[large_angle_inds, :] = \
            angles_large.unsqueeze(1) * axes 

    return phi.squeeze()


def quat_inv(q):
    #input: q: Nx4
    #output: inv(q): Nx4 (conjugate of input quaternions - assumes scalar comes first)

    if q.dim() < 2:
        q = q.unsqueeze(0)

    q_inv = q.clone()
    q_inv[:, 1:] = -q_inv[:, 1:]
     
    return q_inv.squeeze()


def quat_compose(q1, q2):
    #input: q1: Nx4, q2: Nx4
    #output: q1 * q2: Nx4 (composition of two quaternions)

    if q1.dim() < 2:
        q1 = q1.unsqueeze(0)
    if q2.dim() < 2:
        q2 = q2.unsqueeze(0)

    I = torch.diag(q1.new_ones(4)).expand(q1.shape[0], 4, 4)
    q1_w = q1[:, 0].view(q1.shape[0], 1, 1)
    q1_v = q1[:, 1:]
    
    Q1L_a = q1_w * I
    Q1L_b = q1.new_zeros((q1.shape[0], 4, 4))

    Q1L_b[:, 1:, 0] = q1_v
    Q1L_b[:, 0, 1:] = -q1_v
    Q1L_b[:, 1:, 1:] = so3_wedge(q1_v)

    Q1L = Q1L_a + Q1L_b

    q3 = Q1L.bmm(q2.unsqueeze(2))

     
    return q3.squeeze()

def quat_log_diff(q1, q2):
    return quat_log(quat_compose(q1, quat_inv(q2)))


def quat_ang_error(q1, q2):
    log_diff = quat_log_diff(q1, q2)
    if log_diff.dim() < 2:
        return log_diff.norm()
    else:
        return log_diff.norm(dim=1)
    
def batch_logdet3(A):
    if A.dim() < 3:
        A = A.unsqueeze(0)
    try:
        L = torch.cholesky(A)
    except:
        print(A)
    detL = (L[:, 0, 0] * L[:, 1, 1] * L[:, 2, 2])**2
    return torch.log(detL).squeeze()

def nll_quat(q_est, q_gt, Rinv):
    residual = quat_log_diff(q_est, q_gt).unsqueeze(2)
    weighted_term = 0.5*residual.transpose(1,2).bmm(Rinv).bmm(residual)
    nll = -0.5*batch_logdet3(Rinv) + weighted_term.squeeze()
    return  nll

def nll_mat(C_est, C_gt, Rinv):
    residual = so3_log(C_est.bmm(C_gt.transpose(1, 2))).unsqueeze(2)
    weighted_term = 0.5 * residual.transpose(1, 2).bmm(Rinv).bmm(residual)
    nll = weighted_term.squeeze() - 0.5 * batch_logdet3(Rinv)
    return  nll