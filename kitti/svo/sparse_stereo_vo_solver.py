import numpy as np

from liegroups import SE3
from pyslam.sensors import StereoCamera
from pyslam.residuals import PoseResidual, ReprojectionMotionOnlyBatchResidual
from pyslam.problem import Options, Problem
from collections import OrderedDict
from pyslam.losses import L2Loss, HuberLoss, TukeyLoss, TDistributionLoss

class SparseStereoVOSolver(object):
    def __init__(self, camera, obs_stiffness, use_constant_vel_model=False, motion_stiffness=np.identity(6)):
        self.camera = camera
        self.obs_stiffness = obs_stiffness

        # Options
        self.problem_options = Options()
        self.problem_options.allow_nondecreasing_steps = True
        self.problem_options.max_nondecreasing_steps = 3
        self.problem_options.max_iters = 10

        self.problem_solver = Problem(self.problem_options)
        self.solution_key = 'T_cam2_cam1'
        self.params_initial = {self.solution_key: SE3.identity()}

        self.use_constant_vel_model = use_constant_vel_model
        self.motion_stiffness = motion_stiffness
        self.T_21_prev = SE3.identity()

        self.loss = L2Loss()
        # self.loss = HuberLoss(5.)
        # self.loss = TukeyLoss(5.)
        # self.loss = HuberLoss(0.1)
        # self.loss = TDistributionLoss(5.0)  # Kerl et al. ICRA 2013


    def reset_solver(self):
        self.problem_solver = Problem(self.problem_options)

    def set_obs(self, stereo_obs_1, stereo_obs_2, T_21_prev=False):
        self.stereo_obs_1 = stereo_obs_1
        self.stereo_obs_2 = stereo_obs_2
        if T_21_prev:
            self.T_21_prev = T_21_prev


    def set_initial_guess(self, T_21):
        self.params_initial = {self.solution_key: T_21}

    def add_costs(self):
        cost = ReprojectionMotionOnlyBatchResidual(self.camera, self.stereo_obs_1, self.stereo_obs_2, self.obs_stiffness)
        self.problem_solver.add_residual_block(cost, [self.solution_key], loss=self.loss)

        if self.use_constant_vel_model:
            cost = PoseResidual(self.T_21_prev, self.motion_stiffness)
            self.problem_solver.add_residual_block(cost, [self.solution_key], loss=L2Loss())

    def solve(self):
        
        self.add_costs()
        self.problem_solver.initialize_params(self.params_initial)
        self.params_final = self.problem_solver.solve()
        self.problem_solver.compute_covariance()
        return self.params_final[self.solution_key], self.problem_solver._covariance_matrix

#
#
# class SparseStereoVOSolverTranslationOnly(object):
#     def __init__(self, camera, obs_stiffness):
#         self.camera = camera
#         self.obs_stiffness = obs_stiffness
#
#         # Options
#         options = Options()
#         options.allow_nondecreasing_steps = False
#         options.max_nondecreasing_steps = 3
#         #options.linesearch_max_iters = 0
#
#         self.problem_solver = Problem(options)
#         self.solution_key = 't_12_2'
#         self.params_initial = {self.solution_key: np.zeros(3)}
#
#         self.loss = L2Loss()
#         # self.loss = HuberLoss(5.)
#         # self.loss = TukeyLoss(5.)
#         # self.loss = HuberLoss(0.1)
#         # self.loss = TDistributionLoss(5.0)  # Kerl et al. ICRA 2013
#         """Loss function"""
#
#
#     def set_obs(self, stereo_obs_1, stereo_obs_2, C_21):
#         self.stereo_obs_1 = stereo_obs_1
#         self.stereo_obs_2 = stereo_obs_2
#         self.C_21 = C_21
#
#     def set_initial_guess(self, t_12_2):
#         self.params_initial = {self.solution_key: t_12_2}
#
#     def add_costs(self):
#         cost = ReprojectionTranslationOnlyBatchResidual(self.camera, self.stereo_obs_1, self.stereo_obs_2,self.C_21, self.obs_stiffness)
#         self.problem_solver.add_residual_block(cost, [self.solution_key], loss=self.loss)
#
#         # Non-Batch solution
#         # for i in range(len(self.stereo_obs_1)):
#         #     o_1 = self.stereo_obs_1[i]
#         #     o_2 = self.stereo_obs_2[i]
#         #     cost = ReprojectionResidualFrameToFrame(self.camera, o_1, o_2, self.obs_stiffness)
#         #     self.problem_solver.add_residual_block(cost, [self.solution_key])
#
#     def solve(self):
#
#         self.add_costs()
#         self.problem_solver.initialize_params(self.params_initial)
#         self.params_final = self.problem_solver.solve()
#         #print(self.problem_solver.summary(format='brief'))
#
#         return self.params_final[self.solution_key]
