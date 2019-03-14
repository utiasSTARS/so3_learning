import numpy as np

from liegroups import SE3, SO3
from pyslam.problem import Options, Problem
from collections import OrderedDict, namedtuple
from pyslam.losses import L2Loss
from pyslam.residuals import PoseResidual, PoseToPoseResidual, PoseToPoseOrientationResidual
from pyslam.utils import invsqrt

import sys
import time
import pickle


class SO3FusionPipeline(object):
    def __init__(self, T_w_c_vo, Sigma_21_vo,  C_12_hydranet, Sigma_12_hydranet, first_pose=SE3.identity()):
        self.T_w_c = [first_pose] #corrected
        self.T_w_c_vo = T_w_c_vo
        self.Sigma_21_vo = Sigma_21_vo

        self.C_12_hydranet = C_12_hydranet
        self.Sigma_12_hydranet = Sigma_12_hydranet

        assert(Sigma_21_vo.shape[0] == len(self.T_w_c_vo) - 1)
        assert(Sigma_12_hydranet.shape[0] == C_12_hydranet.shape[0])

        self.optimizer = VOFusionSolver()


    def compute_fused_estimates(self):

        start = time.time()
        
        #Start at the second image
        for pose_i in np.arange(1, len(self.T_w_c_vo)):
            self.fuse()

            if pose_i % 100 == 0:
                end = time.time()
                print('Processing pose: {} / {}. Avg. proc. freq.: {:.3f} [Hz]'.format(pose_i, len(self.T_w_c_vo), 100.0/(end - start)))
                start = time.time()

        
    def fuse(self):
        
        pose_i = len(self.T_w_c) - 1
        T_21_vo = self.T_w_c_vo[pose_i+1].inv().dot(self.T_w_c_vo[pose_i])

        #Set initial guess to the corrected guess
        self.optimizer.reset_solver()
        self.optimizer.set_priors(self.T_w_c_vo[pose_i].inv(), self.T_w_c_vo[pose_i+1].inv())

        if np.iscomplex(invsqrt(self.Sigma_12_hydranet[pose_i])).any() or np.linalg.det(self.Sigma_12_hydranet[pose_i]) > 1e-12:
            print('Warning: found bad covariance! invsqrt is complex.')
            print(self.Sigma_12_hydranet[pose_i])
            T_21 = T_21_vo
        else:
            self.optimizer.add_costs(T_21_vo, invsqrt(self.Sigma_21_vo[pose_i]), SO3.from_matrix(self.C_12_hydranet[pose_i], normalize=True), invsqrt(self.Sigma_12_hydranet[pose_i]))
            T_21 = self.optimizer.solve()

        T_w_c = self.T_w_c[-1]
        self.T_w_c.append(T_w_c.dot(T_21.inv()))


class VOFusionSolver(object):
    def __init__(self):

        # Options
        self.problem_options = Options()
        self.problem_options.allow_nondecreasing_steps = True
        self.problem_options.max_nondecreasing_steps = 3
        self.problem_options.max_iters = 10

        self.problem_solver = Problem(self.problem_options)
        self.pose_keys = ['T_1_0', 'T_2_0']
        self.prior_stiffness = invsqrt(1e-12 * np.identity(6))
        self.loss = L2Loss()
        # self.loss = HuberLoss(5.)
        # self.loss = TukeyLoss(5.)
        # self.loss = HuberLoss(0.1)
        # self.loss = TDistributionLoss(5.0)  # Kerl et al. ICRA 2013

    def reset_solver(self):
        self.problem_solver = Problem(self.problem_options)

    def set_priors(self, T_1_0, T_2_0):
        self.params_initial = {self.pose_keys[0]: T_1_0, self.pose_keys[1]: T_2_0}
        prior_residual = PoseResidual(T_1_0, self.prior_stiffness)
        self.problem_solver.add_residual_block(prior_residual, [self.pose_keys[0]])
        self.problem_solver.initialize_params(self.params_initial)

    def add_costs(self, T_21_obs, odom_stiffness, C_21_obs, rot_stiffness):
        residual_pose = PoseToPoseResidual(T_21_obs, odom_stiffness)
        residual_rot = PoseToPoseOrientationResidual(C_21_obs, rot_stiffness)
        self.problem_solver.add_residual_block(residual_pose, self.pose_keys)
        self.problem_solver.add_residual_block(residual_rot, self.pose_keys)

    def solve(self):
        self.params_final = self.problem_solver.solve()
        #self.problem_solver.compute_covariance()
        T_1_0 = self.params_final[self.pose_keys[0]]
        T_2_0 = self.params_final[self.pose_keys[1]]
        T_2_1 = T_2_0.dot(T_1_0.inv())
        return T_2_1