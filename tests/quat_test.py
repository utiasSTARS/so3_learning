from utils import *
from liegroups.torch import SO3
import numpy as np
import time, sys

#Generate random quaternions, take difference and see the result
# num_quat = 100
# q1 = normalize_vecs(torch.rand((num_quat, 4)))
# q2 = normalize_vecs(torch.rand((num_quat, 4)))


# C1 = SO3.from_quaternion(q1)
# C2 = SO3.from_quaternion(q2)

# q3 = quat_compose(q1, quat_inv(q2))
# C3 = C1.dot(C2.inv())

# diff = quat_norms(q3,  C3.to_quaternion()).mean()
# print('average quat diff: {:5.5f}'.format(diff))

# log_diff = (C3.log() - quat_log(q3)).norm(dim=1).mean()

# print('Norm of the log differences: {:5.5f}'.format(log_diff))


# phi = 1e-8*torch.ones(1)
# u = normalize_vecs(torch.rand((3)))

# q = torch.cat((torch.cos(0.5*phi), torch.sin(0.5*phi)*u))
# print(q)
# print(quat_log(q))

A = batch_sample_covariance(torch.randn((16, 30,3)))
end = time.time()
ld = sum([torch.logdet(A[i]) for i in range(A.shape[0])])
print('Completed in {} sec.'.format(time.time() - end))
end = time.time()
ld_cust = batch_logdet3(A).sum()
print('Completed in {} sec.'.format(time.time() - end))
print(ld)
print(ld_cust)
# q1 = normalize_vecs(torch.rand((1, 4)))
# print(q1.norm())
# print(q1)
# print(quat_compose(q1, quat_inv(q1)))

# num_quat = 5
# q = normalize_vecs(torch.rand((num_quat, 4)))
# print(quat_ang_error(q, -q))

# phi1 = (np.pi)*torch.ones(1)
# u = normalize_vecs(torch.rand((3)))
# q1 = torch.cat((torch.cos(0.5*phi1), torch.sin(0.5*phi1)*u))
# phi2 = (1e-6)*torch.ones(1)
# q2 = torch.cat((torch.cos(0.5*phi2), torch.sin(0.5*phi2)*u))

# print(q1)
# print(q2)
# print(quat_ang_error(q1, q2))
# print(quat_norm_diff(q1, q2))

# print(q)
# print(quat_log(q))


##
