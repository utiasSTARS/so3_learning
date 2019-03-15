import numpy as np

from liegroups import SE3, SO3

from pyslam.sensors import StereoCamera
from pyslam.problem import Options, Problem

from collections import OrderedDict, namedtuple

# Optional imports
import cv2
#import pyopengv

import sys
import viso2
import time

from sparse_stereo_vo_solver import SparseStereoVOSolver
from outlier_rejection import FrameToFrameRANSAC
import pickle

from extract_helpers import StereoFeatureTracks


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

class SparseMatcherParams(object):
    def __init__(self):
        self.match_num = 200
        self.matcher_type = 'viso2'        

class SparseStereoPipeline(object):
    def __init__(self, pipeline_params=SparseStereoPipelineParams()):
        self.camera = pipeline_params.camera
        """Camera model"""
        self.T_w_c = [pipeline_params.first_pose]
        """List of camera poses"""
        self.Sigma = []
        """List of relative pose uncertainty"""
        self.params = pipeline_params
        """Pipeline parameters"""

        matcher_params = SparseMatcherParams() 
        self.matcher = SparseStereoQuadMatcher(matcher_params, self.camera)
        self.ransac_obj = FrameToFrameRANSAC(self.camera)

        #Images for tracking
        self.img1_l = []
        self.img1_r = []
        self.img2_l = []
        self.img2_r = []

        #Or saved tracks!
        if self.params.saved_stereo_tracks_file is not None:
            with open(self.params.saved_stereo_tracks_file, 'rb') as f:
                print('Loading saved tracks from: {}.'.format(f))
                self.saved_tracks = pickle.load(f)
        
    def push_back(self, im_left, im_right):
        self.img1_l = self.img2_l
        self.img1_r = self.img2_r
        self.img2_l = im_left
        self.img2_r = im_right
    
    def _convert_to_camera_vectors(self):
        """Convert pixel locations in the left cameras of each pose to unit vectors"""
        
        cu, cv, fu, fv, b = self.camera.cu, self.camera.cv, self.camera.fu, self.camera.fv, self.camera.b
        stereo_obs_1 = np.atleast_2d(self.matcher.matched_stereo_obs_1)
        stereo_obs_2 = np.atleast_2d(self.matcher.matched_stereo_obs_2)


        x_prime_1 = (stereo_obs_1[:,0] - cu)/fu
        y_prime_1 = (stereo_obs_1[:,1] - cv)/fv
        z_prime_1 = np.sqrt(1 - np.square(x_prime_1) - np.square(y_prime_1))

        x_prime_2 = (stereo_obs_2[:,0] - cu)/fu
        y_prime_2 = (stereo_obs_2[:,1] - cv)/fv
        z_prime_2 = np.sqrt(1 - np.square(x_prime_2) - np.square(y_prime_2))

        cam_vec_1 = np.vstack((x_prime_1,y_prime_1,z_prime_1)).T
        cam_vec_2 = np.vstack((x_prime_2,y_prime_2,z_prime_2)).T
        
        return (cam_vec_1, cam_vec_2)

    def apply_blur(self, img):
        return cv2.GaussianBlur(img, (25, 25), )

    def compute_vo(self, dataset):

        start = time.time()
        disp_freq = 100

        #Using RGB images
        #Track features in real time
        if self.params.saved_stereo_tracks_file is None:
            for pose_i, impair in enumerate(dataset.rgb):
                img_l = np.array(impair[0].convert('L'))
                img_r = np.array(impair[1].convert('L'))

                if self.params.apply_gaussian_blur:
                    img_l = self.apply_blur(img_l)
                    img_r = self.apply_blur(img_r)

                self.push_back(img_l, img_r)

                #Wait until the second pose to begin tracking
                if pose_i == 0:
                    continue
                self.track()
                if pose_i % disp_freq == 0:
                    end = time.time()
                    print('Processing {}. Pose: {} / {}. Avg. proc. freq.: {:.3f} [Hz]'.format(self.params.dataset_date_drive, pose_i, len(dataset), disp_freq/(end - start)))
                    start = time.time()
        #Used saved tracks (fast, but requires pre-computing the tracks)
        else:
            #Start at the second image
            for pose_i in np.arange(1, len(dataset)):
                self.track()

                if pose_i % disp_freq == 0:
                    end = time.time()
                    print('Processing {}. Pose: {} / {}. Avg. proc. freq.: {:.3f} [Hz]'.format(self.params.dataset_date_drive, pose_i, len(dataset), disp_freq/(end - start)))
                    start = time.time()

        
    def track(self):
        
        #Normal, real-time matching
        if self.params.saved_stereo_tracks_file is None:
            self.matcher.stereo_quad_match(self.img1_l, self.img1_r, self.img2_l, self.img2_r)

            #RANSAC: select best observations
            self.ransac_obj.set_obs(self.matcher.matched_stereo_obs_1, self.matcher.matched_stereo_obs_2)
            (T_21_guess, stereo_obs_1_inliers, stereo_obs_2_inliers, inlier_mask_best) = self.ransac_obj.perform_ransac()
        #Using a file with saved tracks
        else:
            pose_i = len(self.T_w_c) - 1
            matched_stereo_obs_1 = self.saved_tracks.stereo_obs_1_list[pose_i]
            matched_stereo_obs_2 = self.saved_tracks.stereo_obs_2_list[pose_i]

            #RANSAC: select best observations
            self.ransac_obj.set_obs(matched_stereo_obs_1, matched_stereo_obs_2)
            (T_21_guess, stereo_obs_1_inliers, stereo_obs_2_inliers, inlier_mask_best) = self.ransac_obj.perform_ransac()

        optimizer = SparseStereoVOSolver(self.camera, self.params.obs_stiffness, self.params.use_constant_velocity_motion_model, self.params.motion_stiffness)
        optimizer.set_initial_guess(T_21_guess)

        if self.params.use_constant_velocity_motion_model and len(self.T_w_c) > 1:
            optimizer.set_obs(stereo_obs_1_inliers, stereo_obs_2_inliers, self.T_w_c[-1].inv().dot(self.T_w_c[-2]))
        else:
            optimizer.set_obs(stereo_obs_1_inliers, stereo_obs_2_inliers)

        T_21, Sigma_21 = optimizer.solve()
        T_w_c = self.T_w_c[-1]
        self.T_w_c.append(T_w_c.dot(T_21.inv()))
        self.Sigma.append(Sigma_21)

        

class SparseStereoQuadMatcher(object):
    def __init__(self, match_options, camera):
        self.match_options = match_options
        self.matched_stereo_obs_1 = []
        self.matched_stereo_obs_2 = []

        if self.match_options.matcher_type == 'viso2':
            params = viso2.Stereo_parameters()
            params.calib.f  = camera.fu
            params.calib.cu = camera.cu
            params.calib.cv = camera.cv
            params.base     = camera.b
            matcher_params = viso2.Matcher_parameters()
            matcher_params.nms_n = 20 
            matcher_params.nms_tau = 50
            matcher_params.match_binsize = 50 
            matcher_params.match_radius = 1000
            matcher_params.match_disp_tolerance = 5
            matcher_params.outlier_flow_tolerance = 50
            matcher_params.outlier_disp_tolerance = 50
            matcher_params.multi_stage = 1
            matcher_params.half_resolution = 0
            matcher_params.refinement = 1

            self.matcher = viso2.Matcher(matcher_params)
            self.matcher.setIntrinsics(params.calib.f, params.calib.cu, params.calib.cv, params.base)

        self.match_count = 0

    def stereo_quad_match(self, img1_l, img1_r, img2_l, img2_r):
        self.img1_l = img1_l
        self.img1_r = img1_r
        self.img2_l = img2_l
        self.img2_r = img2_r

        if self.match_options.matcher_type == 'simple_orb':
            self._stereo_quad_match_simple()
        elif self.match_options.matcher_type == 'viso2':
            self._stereo_quad_match_viso2()
        else:
            self._stereo_quad_match_circle()
        
        #Prune all observations with disparity close to 0
        self._prune_stereo_obs()
        #print('Matches after pruning: {} '.format(len(self.matched_stereo_obs_1)))

    
    """A matcher based on libviso2"""
    def _stereo_quad_match_viso2(self):
        if self.match_count < 1:
            self.matcher.pushBack(self.img1_l, self.img1_r)

        self.match_count += 1
        self.matcher.pushBack(self.img2_l, self.img2_r)
        self.matcher.matchFeatures(2)

        matches = self.matcher.getMatches()
        #print('VISO2 matched {} features.'.format(matches.size()))
        
        self.matched_stereo_obs_1 = [np.array([m.u1p, m.v1p, m.u1p - m.u2p]) for m in matches]
        self.matched_stereo_obs_2 = [np.array([m.u1c, m.v1c, m.u1c - m.u2c]) for m in matches]


       


    """A simpler matcher that only uses the left images in each stereo pair to track temporally"""
    def _stereo_quad_match_simple(self):
        #Orb extractor
        ft = cv2.ORB_create()
        
        # find the keypoints and descriptors with ORB
        kp1_l, des1_l = ft.detectAndCompute(self.img1_l,None)
        kp1_r, des1_r = ft.detectAndCompute(self.img1_r,None)
        kp2_l, des2_l = ft.detectAndCompute(self.img2_l,None)
        kp2_r, des2_r = ft.detectAndCompute(self.img2_r,None)

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        #Simple left-right and left1-left2 matching

        #l -> r
        (m_ids_1l_1, m_ids_1r_1) = self._match_pair(bf, des1_l, des1_r, sort_and_select_top = True, top_num_matches = self.match_options.match_num)
        (m_ids_2l_1, m_ids_2r_1) = self._match_pair(bf, des2_l, des2_r, sort_and_select_top = True, top_num_matches = self.match_options.match_num)
    
        des1_l_matched = des1_l[m_ids_1l_1,:]
        des2_l_matched = des2_l[m_ids_2l_1,:]
        
        
        #1l -> 2l
        (m_1l_mask, m_2l_mask) = self._match_pair(bf, des1_l_matched, des2_l_matched, sort_and_select_top = True, top_num_matches = self.match_options.match_num)
        
        match_ids_1l = m_ids_1l_1[m_1l_mask]
        match_ids_1r = m_ids_1r_1[m_1l_mask]
        match_ids_2l = m_ids_2l_1[m_2l_mask]
        match_ids_2r = m_ids_2r_1[m_2l_mask]

        # Convert to uvd

        kpt1_l_matched = [kp1_l[i] for i in match_ids_1l]
        kpt1_r_matched = [kp1_r[i] for i in match_ids_1r]
        kpt2_l_matched = [kp2_l[i] for i in match_ids_2l]
        kpt2_r_matched = [kp2_r[i] for i in match_ids_2r]
        

        self.matched_stereo_obs_1 = self._convert_uv_keypoints_to_uvd(kpt1_l_matched, kpt1_r_matched)
        self.matched_stereo_obs_2 = self._convert_uv_keypoints_to_uvd(kpt2_l_matched, kpt2_r_matched)
        
    """Inspired by Libviso2, this tracker matches a stereo quad (i.e. consecutive stereo pairs) by tracking features in a 'circle' that starts at the first left image and only keeps features that survive the round trip around the stereo quad """
    def _stereo_quad_match_circle(self):
        
        #Orb extractor
        ft = cv2.ORB_create()
        
        # find the keypoints and descriptors with ORB
        kp1_l, des1_l = ft.detectAndCompute(self.img1_l,None)
        kp1_r, des1_r = ft.detectAndCompute(self.img1_r,None)
        kp2_l, des2_l = ft.detectAndCompute(self.img2_l,None)
        kp2_r, des2_r = ft.detectAndCompute(self.img2_r,None)

        # create BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        #We will match in a square, 1l->1r->2r->2l->1l and keep only those keypoints that survive the entire journey

        #1l -> 1r
        (m_ids_1l_1, m_ids_1r_1) = self._match_pair(bf, des1_l, des1_r, sort_and_select_top = True, top_num_matches = self.match_options.match_num)
        print('1l -> 1r: matched {} '.format(len(m_ids_1l_1)))
    

        #1r -> 2r
        (m_ids_1r_2, m_ids_2r_1) = self._match_pair(bf, des1_r[m_ids_1r_1], des2_r)
        m_ids_1r_matched = (np.arange(len(des1_r))[m_ids_1r_1])[m_ids_1r_2]
        
        print('1r -> 2r: matched {} '.format(len(m_ids_1r_2)))

        #2r -> 2l
        (m_ids_2r_2, m_ids_2l_1) = self._match_pair(bf, des2_r[m_ids_2r_1], des2_l)
        m_ids_2r_matched = (np.arange(len(des2_r))[m_ids_2r_1])[m_ids_2r_2]
        
        print('2r -> 2l: matched {} '.format(len(m_ids_2r_2)))
        
        #2l -> 1l
        (m_ids_2l_2, m_ids_1l_2) = self._match_pair(bf, des2_l[m_ids_2l_1], des1_l)
        m_ids_2l_matched = (np.arange(len(des2_l))[m_ids_2l_1])[m_ids_2l_2]

        print('2l -> 1l: matched {} '.format(len(m_ids_1l_2)))
    
        
        
        match_ids_1l = [m_ids_1l_1[i] for i in range(len(m_ids_1l_1)) if m_ids_1l_1[i] in m_ids_1l_2]
        match_ids_1r = [m_ids_1r_1[i] for i in range(len(m_ids_1r_1)) if m_ids_1l_1[i] in m_ids_1l_2]
        
        print('Matches that pass consistency: {} '.format(len(match_ids_1l)))


        mask1 = [i for i in range(len(m_ids_1l_2)) if m_ids_1l_2[i] in match_ids_1l]
        match_ids_2l = m_ids_2l_matched[mask1]
        
        mask2 = [i for i in range(len(m_ids_2l_1)) if m_ids_2l_1[i] in match_ids_2l]
        match_ids_2r = m_ids_2r_matched[mask2]
    
        # print(len(match_ids_1l))
        # print(len(match_ids_1r))
        # print(len(match_ids_2l))
        # print(len(match_ids_2r))
        
        kpt1_l_matched = [kp1_l[i] for i in match_ids_1l]
        kpt1_r_matched = [kp1_r[i] for i in match_ids_1r]
        kpt2_l_matched = [kp2_l[i] for i in match_ids_2l]
        kpt2_r_matched = [kp2_r[i] for i in match_ids_2r]
        

        # Convert to uvd
        self.stereo_obs_1 = self._convert_uv_keypoints_to_uvd(kpt1_l_matched, kpt1_r_matched)
        self.stereo_obs_2 = self._convert_uv_keypoints_to_uvd(kpt2_l_matched, kpt2_r_matched)

        # matches = namedtuple('Matches', ['queryIdx', 'trainIdx'])
        # matches.queryIdx = match_ids_1l
        # matches.trainIdx = match_ids_2l

        #img3 = cv2.drawMatches(img1_l,kpt1_l_matched,img2_l,kpt2_l_matched, matches,None, flags=2)
        #plt.imshow(img3),plt.show()


    def _match_pair(self, matcher, des1, des2, sort_and_select_top = False, top_num_matches = None):
    
        matches = matcher.match(des1, des2)

        if sort_and_select_top:
            matches = sorted(matches, key = lambda x:x.distance)
            # Take best N matches
            matches = matches[0:top_num_matches]

        m_ids1 = np.array([matches[i].queryIdx for i in range(len(matches))])
        m_ids2 = np.array([matches[i].trainIdx for i in range(len(matches))])

        return (m_ids1, m_ids2)

    def _convert_uv_keypoints_to_uvd(self, kp_l, kp_r):
        return [np.array([kp_l[i].pt[0], kp_l[i].pt[1], kp_l[i].pt[0] -  kp_r[i].pt[0]]) for i in range(len(kp_l))]

    def _prune_stereo_obs(self):
        """Ensure that disparity is not 0"""
        del_indices = [i for i in range(len(self.matched_stereo_obs_1)) if self.matched_stereo_obs_1[i][2] < 0.01 or self.matched_stereo_obs_2[i][2] < 0.01]

        for i in sorted(del_indices, reverse=True):
            self.matched_stereo_obs_1.pop(i)
            self.matched_stereo_obs_2.pop(i)
    
      





    






