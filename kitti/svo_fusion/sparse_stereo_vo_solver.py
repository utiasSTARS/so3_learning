import numpy as np

from liegroups import SE3
from pyslam.sensors import StereoCamera
from pyslam.residuals import PoseResidual, ReprojectionMotionOnlyBatchResidual, PoseOrientationResidual
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
        self.reprojection_loss = TDistributionLoss(5.0)  # Kerl et al. ICRA 2013



    def reset_solver(self):
        self.problem_solver = Problem(self.problem_options)

    def set_obs(self, stereo_obs_1, stereo_obs_2, T_21_prev=False):
        self.stereo_obs_1 = stereo_obs_1
        self.stereo_obs_2 = stereo_obs_2
        if T_21_prev:
            self.T_21_prev = T_21_prev
        
    def set_initial_guess(self, T_21):
        self.params_initial = {self.solution_key: T_21}

    def add_orientation_residual(self, C_21_obs, stiffness):
        residual_rot = PoseOrientationResidual(C_21_obs, stiffness)
        self.problem_solver.add_residual_block(residual_rot, [self.solution_key], loss=self.loss)

    def add_costs(self):
        cost = ReprojectionMotionOnlyBatchResidual(self.camera, self.stereo_obs_1, self.stereo_obs_2, self.obs_stiffness)
        self.problem_solver.add_residual_block(cost, [self.solution_key], loss=self.reprojection_loss)

    def solve(self):
        
        self.add_costs()
        self.problem_solver.initialize_params(self.params_initial)
        self.params_final = self.problem_solver.solve()
        self.problem_solver.compute_covariance()
        return self.params_final[self.solution_key], self.problem_solver._covariance_matrix
