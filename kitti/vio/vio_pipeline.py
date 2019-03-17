import numpy as np

from liegroups.numpy import SE3, SO3
from pyslam.problem import Options, Problem
from collections import OrderedDict, namedtuple
from pyslam.losses import L2Loss, TDistributionLoss, HuberLoss
from pyslam.residuals import PoseResidual, PoseToPoseResidual, PoseToPoseOrientationResidual
from pyslam.utils import invsqrt
import torch
import sys
import time
import pickle
import copy

class VisualInertialPipeline():
    def __init__(self, dataset, T_cam_imu, hydranet_output_file, first_pose=SE3.identity()):
        self.optimizer = PoseFusionSolver()
        self.dataset = dataset
        self.dataset._load_timestamps()
        self.imu_Q = self.compute_imu_Q()
        self.T_c_w = [first_pose]
        self.T_c_w_imu = [first_pose]

        self.T_cam_imu = T_cam_imu

        self._load_hydranet_files(hydranet_output_file)

        self.T_c_w_gt = [T_cam_imu.dot(SE3.from_matrix(o.T_w_imu).inv())
                         for o in self.dataset.oxts]

        self.pose_skip = 2 #0 means apply every delta

    def _load_hydranet_files(self, path):
        hn_data = torch.load(path)
        self.Sigma_21_hydranet = hn_data['Sigma_21'].numpy()
        self.C_21_hydranet = hn_data['Rot_21'].numpy()
        self.C_21_hydranet_gt = hn_data['Rot_21_gt'].numpy()
        self.Sigma_21_hydranet_const = self.compute_rot_covar()
        self.C_21_large_err_mask = self.compute_large_err_mask()

    def compute_large_err_mask(self):
        phi_errs = np.empty((len(self.C_21_hydranet_gt)))
        for i in range(len(self.C_21_hydranet_gt)):
            C_21_est = SO3.from_matrix(self.C_21_hydranet[i], normalize=True)
            C_21_gt = SO3.from_matrix(self.C_21_hydranet_gt[i], normalize=True)
            phi_errs[i] = C_21_est.dot(C_21_gt.inv()).log()[1]
        return phi_errs > 0.2*np.pi/180.

    def compute_rot_covar(self):
        phi_errs = np.empty((len(self.C_21_hydranet_gt), 3))
        for i in range(len(self.C_21_hydranet_gt)):
            C_21_est = SO3.from_matrix(self.C_21_hydranet[i], normalize=True)
            C_21_gt = SO3.from_matrix(self.C_21_hydranet_gt[i], normalize=True)
            phi_errs[i] = C_21_est.dot(C_21_gt.inv()).log()

        print(np.mean(phi_errs, axis=0)*180/np.pi)
        return np.cov(phi_errs, rowvar=False)


    def compute_imu_Q(self):
        T_w_imu_gt = [SE3.from_matrix(o.T_w_imu) for o in self.dataset.oxts]

        xi_errs = np.empty((len(self.dataset.oxts) - 1, 6))
        for pose_i, oxt in enumerate(self.dataset.oxts):

            if pose_i == len(self.dataset.oxts) - 1:
                break
            dt = (self.dataset.timestamps[pose_i + 1] - self.dataset.timestamps[pose_i]).total_seconds()
            xi = -dt*self._assemble_motion_vec(oxt)
            T_21_imu = SE3.exp(xi)
            T_21_gt = T_w_imu_gt[pose_i+1].inv().dot(T_w_imu_gt[pose_i])
            xi_errs[pose_i] = T_21_imu.dot(T_21_gt.inv()).log()/(dt)

        return np.cov(xi_errs, rowvar=False)


    def _assemble_motion_vec(self, oxt):
        motion_vec = np.empty(6)
        motion_vec[0] = oxt.packet.vf
        motion_vec[1] = oxt.packet.vl
        motion_vec[2] = oxt.packet.vu
        motion_vec[3] = oxt.packet.wx
        motion_vec[4] = oxt.packet.wy
        motion_vec[5] = oxt.packet.wz
        return motion_vec

    poses_skipped = 0
    def compute_vio(self):
        num_poses = 100#len(self.dataset.oxts)
        start = time.time()

        for pose_i, oxt in enumerate(self.dataset.oxts[:num_poses]):
            if pose_i == num_poses - 1:
                break

            if pose_i % 100 == 0:
                end = time.time()
                print('Processing pose: {} / {}. Avg. proc. freq.: {:.3f} [Hz]'.format(pose_i, len(self.dataset.oxts),100.0/(end - start)))
                start = time.time()


            dt = (self.dataset.timestamps[pose_i+1] - self.dataset.timestamps[pose_i]).total_seconds()
            xi = -dt*self._assemble_motion_vec(oxt)
            T_21_imu = self.T_cam_imu.dot(SE3.exp(xi)).dot(self.T_cam_imu.inv())

            Ad_T_cam_imu = SE3.adjoint(self.T_cam_imu)
            Sigma_21_imu = Ad_T_cam_imu.dot(dt*dt*self.imu_Q).dot(Ad_T_cam_imu.transpose())

            Sigma_hn = self.Sigma_21_hydranet[pose_i]
            #Sigma_hn[1,1] = 500*Sigma_hn[1,1]
            #Sigma_hn = self.Sigma_21_hydranet_const

            C_21_hn = SO3.from_matrix(self.C_21_hydranet[pose_i])
            #C_21_gt = SO3.from_matrix(self.C_21_hydranet_gt[pose_i], normalize=True)
            #rot_err = C_21_hn.dot(C_21_gt.inv()).log()

            self.optimizer.reset_solver()
            self.optimizer.add_costs(T_21_imu, invsqrt(Sigma_21_imu), C_21_hn, invsqrt(Sigma_hn))


            self.optimizer.set_priors(self.T_c_w[-1], T_21_imu.dot(self.T_c_w[-1]))

            # if self.C_21_large_err_mask[pose_i]:
            #     T_21 = copy.deepcopy(T_21_imu)
            # else:

            T_c_w = self.optimizer.solve()

            # T_21_gt = self.T_c_w_gt[pose_i + 1].dot(self.T_c_w_gt[pose_i].inv())
            # T_c_w = T_21_gt.dot(self.T_c_w[-1])


            #
            #solved_err = np.linalg.norm(C_21.dot(C_21_gt.inv()).log())
            # imu_err =  np.linalg.norm(T_21_imu.dot(T_21_gt.inv()).log())
            #
            # if solved_err > imu_err:
            #     print('IMU -> Solved error: {:.5E} -> {:.5E}'.format(imu_err, solved_err))
            #     T_21 = T_21_imu

            self.T_c_w.append(T_c_w)
            self.T_c_w_imu.append(T_21_imu.dot(self.T_c_w_imu[-1]))


class PoseFusionSolver(object):
    def __init__(self):

        # Options
        self.problem_options = Options()
        self.problem_options.allow_nondecreasing_steps = False
        self.problem_options.max_nondecreasing_steps = 3
        self.problem_options.max_iters = 10

        self.problem_solver = Problem(self.problem_options)
        self.pose_keys = ['T_1_0', 'T_2_0']
        self.loss = L2Loss()
        # self.loss = HuberLoss(5.)
        # self.loss = TukeyLoss(5.)
        # self.loss = HuberLoss(.1)
        #self.loss = HuberLoss(10.0)  # Kerl et al. ICRA 2013

    def reset_solver(self):
        self.problem_solver = Problem(self.problem_options)

    def set_priors(self, T_1_0, T_2_0):
        self.params_initial = {self.pose_keys[0]: T_1_0, self.pose_keys[1]: T_2_0}
        self.problem_solver.set_parameters_constant(self.pose_keys[0])
        self.problem_solver.initialize_params(self.params_initial)

    def add_costs(self, T_21_obs, odom_stiffness, C_21_obs, rot_stiffness):
        residual_pose = PoseToPoseResidual(T_21_obs, odom_stiffness)
        residual_rot = PoseToPoseOrientationResidual(C_21_obs, rot_stiffness)
        self.problem_solver.add_residual_block(residual_pose, self.pose_keys)
        self.problem_solver.add_residual_block(residual_rot, self.pose_keys, loss=self.loss)

    def solve(self):
        self.params_final = self.problem_solver.solve()
        #print(self.problem_solver.summary())
        #self.problem_solver.compute_covariance()
        T_1_0 = self.params_final[self.pose_keys[0]]
        T_2_0 = self.params_final[self.pose_keys[1]]
        #T_2_1 = T_2_0.dot(T_1_0.inv())
        return T_2_0