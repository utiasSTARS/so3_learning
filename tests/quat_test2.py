from liegroups.torch import SO3
import numpy as np
import torch
import time, sys, math
sys.path.insert(0,'..')
from utils import *

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
#

def allclose(mat1, mat2, tol=1e-6):
    """Check if all elements of two tensors are close within some tolerance.

    Either tensor can be replaced by a scalar.
    """
    return isclose(mat1, mat2, tol).all()


def isclose(mat1, mat2, tol=1e-6):
    """Check element-wise if two tensors are close within some tolerance.

    Either tensor can be replaced by a scalar.
    """
    return (mat1 - mat2).abs_().lt(tol)


# C = torch.tensor([[ 0.8638,  0.0675, -0.4993],
#          [ 0.0676, -0.9976, -0.0180],
#          [-0.4993, -0.0182, -0.8662]])
# C = SO3.from_matrix(C, normalize=True).as_matrix()
# print(quaternion_from_matrix(C.numpy()))
# print(SO3.from_matrix(C, normalize=True).to_quaternion())

phi_mat = 3.14*torch.rand((5000,1))*normalize_vecs(torch.randn((5000,3))) #Ensure max phi.norm() < pi
q = quat_exp(phi_mat)
phi_check = quat_log(q)
if not allclose(phi_mat, phi_check):

    far_ids = (phi_mat - phi_check).abs().gt(1e-6).nonzero()[:,0].unique()
    print(far_ids.shape[0])
    print(phi_mat[far_ids])
    print(phi_mat[far_ids].norm(dim=1))

    #print(phi_mat[(phi_mat - phi_check).abs().gt(1e-6)].view(-1, 3))
    #print(phi_check[(phi_mat - phi_check).abs().gt(1e-6)].view(-1, 3))
else:
    print('All close.')

q = normalize_vecs(torch.randn((5, 4)))
print(perturb_quat_for_hydranet(q, 2, 0.))


