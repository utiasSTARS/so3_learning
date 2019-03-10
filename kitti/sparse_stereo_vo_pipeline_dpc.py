import numpy as np

from liegroups import SE3, SO3

from pyslam.sensors import StereoCamera
from pyslam.problem import Options, Problem

from collections import OrderedDict, namedtuple

# Optional imports
#import cv2
#import pyopengv

import sys
import time
from outlier_rejection import FrameToFrameRANSAC
import pickle
from extract_helpers import StereoFeatureTracks
from sparse_stereo_vo_solver import SparseStereoVOSolverTranslationOnly

class SparseStereoPipelineParams(object):
    def __init__(self):
        self.camera = []
        self.first_pose = SE3.identity()
        self.obs_stiffness = []
        self.optimize_trans_only = False
        self.use_constant_velocity_motion_model = False
        self.motion_stiffness = []
        self.dataset_date_drive = ''
        self.saved_stereo_tracks_file = None

class SparseStereoPipelineDPC(object):
    def __init__(self, T_w_c_init_corr, pipeline_params=SparseStereoPipelineParams()):
        self.camera = pipeline_params.camera
        """Camera model"""
        self.T_w_c = [pipeline_params.first_pose]
        """List of camera poses"""
        self.T_w_c_init_corr = T_w_c_init_corr
        """List of corrected poses from the Deep Pose Corrector"""
        
        self.params = pipeline_params
        """Pipeline parameters"""

        self.ransac_obj = FrameToFrameRANSAC(self.camera)

        #Use saved tracks
        with open(self.params.saved_stereo_tracks_file, 'rb') as f:
            self.saved_tracks = pickle.load(f)


    def compute_vo(self, dataset):

        start = time.time()
        
        #Start at the second image
        for pose_i in np.arange(1, len(dataset)):
            self.track()

            if pose_i % 100 == 0:
                end = time.time()
                print('Processing {}. Pose: {} / {}. Avg. proc. freq.: {:.3f} [Hz]'.format(self.params.dataset_date_drive, pose_i, len(dataset), 100.0/(end - start)))
                start = time.time()

        
    def track(self):
        
        pose_i = len(self.T_w_c) - 1
        matched_stereo_obs_1 = self.saved_tracks.stereo_obs_1_list[pose_i]
        matched_stereo_obs_2 = self.saved_tracks.stereo_obs_2_list[pose_i]

        #RANSAC: select best observations
        self.ransac_obj.set_obs(matched_stereo_obs_1, matched_stereo_obs_2)
        (_, stereo_obs_1_inliers, stereo_obs_2_inliers, inlier_mask_best) = self.ransac_obj.perform_ransac()

        #optimizer = SparseStereoVOSolver(self.camera, self.params.obs_stiffness, self.params.use_constant_velocity_motion_model, self.params.motion_stiffness)
        optimizer = SparseStereoVOSolverTranslationOnly(self.camera, self.params.obs_stiffness)


        #Set initial guess to the corrected guess
        T_21_dpc_corr = self.T_w_c_init_corr[pose_i+1].inv().dot( self.T_w_c_init_corr[pose_i])
        
        #optimizer.set_initial_guess(T_21_guess)
        #optimizer.set_obs(stereo_obs_1_inliers, stereo_obs_2_inliers)

        optimizer.set_initial_guess(T_21_dpc_corr.trans)
        optimizer.set_obs(stereo_obs_1_inliers, stereo_obs_2_inliers, T_21_dpc_corr.rot)
        

        t_12_2 = optimizer.solve()
        T_21 = SE3(rot=T_21_dpc_corr.rot, trans=t_12_2)
        T_w_c = self.T_w_c[-1]
        self.T_w_c.append(T_w_c.dot(T_21.inv()))