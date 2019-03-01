import pykitti
#import cv2
import numpy as np
from liegroups import SE3, SO3
from pyslam.sensors import StereoCamera
from pyslam.utils import invsqrt
from pyslam.metrics import TrajectoryMetrics
from optparse import OptionParser
from sparse_stereo_vo_pipeline import SparseMatcherParams, SparseStereoQuadMatcher
import time
import pickle
import os
from extract_helpers import StereoImageHelper, StereoFeatureTracks

def extract_tracks(basedir, outdir, date, drive, im_range):

    #Load data
    dataset = pykitti.raw(basedir, date, drive, frames=im_range, imformat='cv2')
    dataset_len = len(dataset)

    #Setup KITTI Camera parameters
    test_im = next(dataset.cam0)
    fu = dataset.calib.K_cam0[0, 0]
    fv = dataset.calib.K_cam0[1, 1]
    cu = dataset.calib.K_cam0[0, 2]
    cv = dataset.calib.K_cam0[1, 2]
    b = dataset.calib.b_gray
    h, w = test_im.shape
    kitti_camera0 = StereoCamera(cu, cv, fu, fv, b, w, h)
    
    
    matcher_params = SparseMatcherParams() 
    matcher = SparseStereoQuadMatcher(matcher_params, kitti_camera0)
    

    #Save all the matched_stereo_obs in a file
    saved_tracks = StereoFeatureTracks()
    img_helper = StereoImageHelper()

    start = time.time()
    for pose_i, impair in enumerate(dataset.gray):

        img_l = impair[0]
        img_r = impair[1]

        img_helper.push_back(img_l, img_r)

        #Wait until the second pose to begin tracking
        if pose_i == 0:
            continue
        
        matcher.stereo_quad_match(img_helper.img1_l, img_helper.img1_r, img_helper.img2_l, img_helper.img2_r)
        saved_tracks.append(matcher.matched_stereo_obs_1, matcher.matched_stereo_obs_2)
        
        if pose_i % 10 == 0:
            end = time.time()
            print('Processing '+date+'_'+drive+'. Pose: {} / {}. Avg. processing freq: {:.3f} [Hz]'.format(pose_i, len(dataset), 10.0/(end - start)))
            start = time.time()
    
    if im_range is not None:
        file_name = '{}_{}_frames_{}-{}_saved_tracks.pickle'.format(date, drive, im_range[0], im_range[-1])
    else:
        file_name = '{}_{}_frames_{}-{}_saved_tracks.pickle'.format(date, drive, 0, len(dataset))

    
    out_file = os.path.join(outdir, file_name)

    print('Saving to pickle file: {}'.format(out_file))
    with open(out_file, 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(saved_tracks, f, pickle.HIGHEST_PROTOCOL)


#This lovely code was shamelessly stolen from Lee's dense stereo example
def main():
    # Odometry sequences
    # Nr.     Sequence name     Start   End
    # ---------------------------------------
    # 00: 2011_10_03_drive_0027 000000 004540
    # 01: 2011_10_03_drive_0042 000000 001100
    # 02: 2011_10_03_drive_0034 000000 004660
    # 03: 2011_09_26_drive_0067 000000 000800
    # 04: 2011_09_30_drive_0016 000000 000270
    # 05: 2011_09_30_drive_0018 000000 002760
    # 06: 2011_09_30_drive_0020 000000 001100
    # 07: 2011_09_30_drive_0027 000000 001100
    # 08: 2011_09_30_drive_0028 001100 005170
    # 09: 2011_09_30_drive_0033 000000 001590
    # 10: 2011_09_30_drive_0034 000000 001200

    #Dataset
    basedir = '/media/m2-drive/datasets/KITTI/distorted_images'
    outdir = '/media/raid5-array/datasets/KITTI/extracted_sparse_tracks/distorted'
    #basedir = '/Users/valentin/Research/KITTI'
    #outdir = '/Users/valentin/Research/KITTI/saved_tracks'

    os.makedirs(outdir, exist_ok=True)

    seqs = {'00': {'date': '2011_10_03',
                   'drive': '0027',
                   'frames': range(0, 4541)},
            '01': {'date': '2011_10_03',
                   'drive': '0042',
                   'frames': range(0, 1101)},
            '02': {'date': '2011_10_03',
                   'drive': '0034',
                   'frames': range(0, 4661)},
            '04': {'date': '2011_09_30',
                   'drive': '0016',
                   'frames': range(0, 271)},
            '05': {'date': '2011_09_30',
                   'drive': '0018',
                   'frames': range(0, 2761)},
            '06': {'date': '2011_09_30',
                   'drive': '0020',
                   'frames': range(0, 1101)},
            '07': {'date': '2011_09_30',
                   'drive': '0027',
                   'frames': range(0, 1101)},
            '08': {'date': '2011_09_30',
                   'drive': '0028',
                   'frames': range(1100, 5171)},
            '09': {'date': '2011_09_30',
                   'drive': '0033',
                   'frames': range(0, 1591)},
            '10': {'date': '2011_09_30',
                   'drive': '0034',
                   'frames': range(0, 1201)}}

    for key, val in seqs.items():
        date = val['date']
        drive = val['drive']
        frames = val['frames']
        #if key is '03':
        #    continue

        print('Odometry sequence {} | {} {}'.format(key, date, drive))
        #outfile = os.path.join(outdir, date + '_drive_' + drive + '.pickle')
        extract_tracks(basedir, outdir, date, drive, frames)

if __name__ == '__main__':
    main()

