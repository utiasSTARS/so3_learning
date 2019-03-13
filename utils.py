import torch
import numpy as np
import matplotlib.pyplot as plt
import math
from lie_algebra import so3_wedge, so3_log

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

#Ensure all quaternions have positive w
def set_quat_sign(q):

    #This function assumes there a re multiple heads so we have an input of  H x B x 4 (Heads x Batch x 4)
    if q.dim() < 3:
        q = q.unsqueeze(1)

    # Check for negative scalars first, then substitute q for -q whenever that is the case (this accounts for the double cover of S3 over SO(3))
    neg_angle_mask = q[:, :, 0] < 0.
    neg_angle_inds = neg_angle_mask.nonzero().squeeze_()

    if len(neg_angle_inds) > 0:
        q[neg_angle_mask, :] = -1. * q[neg_angle_mask, :]

    return q.squeeze(1)


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

def quat_exp(phi):

    # input: phi: Nx3
    # output: Exp(phi) Nx4 (see Sola eq. 101)

    if phi.dim() < 2:
        phi = phi.unsqueeze(0)

    q = phi.new_empty((phi.shape[0], 4))
    phi_norm = phi.norm(dim=1, keepdim=True)
    q[:,0] = torch.cos(phi_norm.squeeze()/2.)
    q[:, 1:] = (phi/phi_norm)*torch.sin(phi_norm/2.)
    return q.squeeze(0)

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

def positive_fn(x):
    large_num = 10
    y = torch.empty_like(x)
    large_mask = x > large_num
    small_mask = 1 - large_mask
    y[large_mask] = x[large_mask]
    y[small_mask] = torch.log(1. + torch.exp(x[small_mask]))
    return y

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

#input: Nx4
#output: HNx4 where H is num_heads - each target is repeated
def perturb_quat_for_hydranet(q, num_heads, q_sigma):
    if q_sigma > 0.:
        q = q.repeat((num_heads, 1))
        dphi = q_sigma*torch.randn_like(q)[:, :3]
        q = quat_compose(quat_exp(dphi), q)
        return q
    else:
        return q.repeat((num_heads, 1))


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
        print('Cholesky dont work!')
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


#Alternate function because SO3.to_quaternion has some instabilities
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
        print('Cholesky dont work!')
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