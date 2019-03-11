import pykitti
from matplotlib import pyplot as plt
import numpy as np
from liegroups import SE3, SO3
from pyslam.utils import invsqrt
from pyslam.metrics import TrajectoryMetrics
from optparse import OptionParser
from fusion_pipeline import SO3FusionPipeline

import copy
import time
import os, glob
import pickle
import csv

import argparse

parser = argparse.ArgumentParser(description='SO(3) Fusion')
parser.add_argument('--seq', '-s', default='00', type=str,
                    help='which sequence to test')


def run_fusion(basedir, date, drive, im_range, saved_tracks_filename, corrected_metrics_file):

    # Load data
    dataset = pykitti.raw(basedir, date, drive, frames=im_range)


    # Load initial pose and ground truth
    T_cam_imu = SE3.from_matrix(dataset.calib.T_cam2_imu)
    T_cam_imu.normalize()
    T_w_0 = SE3.from_matrix(dataset.oxts[0].T_w_imu).dot(T_cam_imu.inv())
    T_w_0.normalize()
    T_w_c_gt = [SE3.from_matrix(o.T_w_imu).dot(T_cam_imu.inv())
                for o in dataset.oxts]


    tm_corr = TrajectoryMetrics.loadmat(corrected_metrics_file)
    fusion_pipeline = SO3FusionPipeline(tm_corr.Twv_est)
    
    #The magic!
    fusion_pipeline.compute_fused_estimates(dataset)
    
    #Compute statistics
    T_w_c_est = [T for T in fusion_pipeline.T_w_c]
    tm = TrajectoryMetrics(T_w_c_gt, T_w_c_est, convention='Twv')
    # # Save to file
    # if metrics_filename:
    #     print('Saving to {}'.format(metrics_filename))
    #     tm.savemat(metrics_filename)

    return tm




def make_topdown_plot(tm, outfile=None):
    pos_gt = np.array([T.trans for T in tm.Twv_gt])
    pos_est = np.array([T.trans for T in tm.Twv_est])

    f, ax = plt.subplots()

    ax.plot(pos_gt[:, 0], pos_gt[:, 1], '-k',
            linewidth=2, label='Ground Truth')
    ax.plot(pos_est[:, 0], pos_est[:, 1], label='VO')

    ax.axis('equal')
    ax.minorticks_on()
    ax.grid(which='both', linestyle=':', linewidth=0.2)
    ax.set_title('Trajectory')
    ax.set_xlabel('Easting (m)')
    ax.set_ylabel('Northing (m)')
    ax.legend()

    if outfile:
        print('Saving to {}'.format(outfile))
        f.savefig(outfile)

    return f, ax


def make_segment_err_plot(tm, segs, outfile=None):
    segerr, avg_segerr = tm.segment_errors(segs)

    f, ax = plt.subplots(1, 2, figsize=(12, 4))

    ax[0].plot(avg_segerr[:, 0], avg_segerr[:, 1] * 100., '-s')
    ax[1].plot(avg_segerr[:, 0], avg_segerr[:, 2] * 180. / np.pi, '-s')

    ax[0].minorticks_on()
    ax[0].grid(which='both', linestyle=':', linewidth=0.2)
    ax[0].set_title('Translational error')
    ax[0].set_xlabel('Sequence length (m)')
    ax[0].set_ylabel('Average error (\%)')

    ax[1].minorticks_on()
    ax[1].grid(which='both', linestyle=':', linewidth=0.2)
    ax[1].set_title('Rotational error')
    ax[1].set_xlabel('Sequence length (m)')
    ax[1].set_ylabel('Average error (deg/m)')

    if outfile:
        print('Saving to {}'.format(outfile))
        f.savefig(outfile)

    return f, ax


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

    #parse args
    args = parser.parse_args()
    seq = args.seq
    corr_type = args.type


    kitti_basedir = '/media/m2-drive/datasets/KITTI/raw'
    saved_tracks_dir = '/media/raid5-array/datasets/KITTI/extracted_sparse_tracks'
    metrics_path = '/media/raid5-array/experiments/Deep-PC/stereo_vo_results'
    
    corrected_metrics_file = glob.glob(os.path.join(metrics_path, 'corrected/seq_{}_corr_{}_epoch_*.mat'.format(seq,corr_type)))[0]
    orig_metrics_file = os.path.join(metrics_path, 'baseline/{}_drive_{}.mat'.format(seqs[seq]['date'],seqs[seq]['drive']))

    saved_tracks_filename = os.path.join(saved_tracks_dir, '{}_{}_frames_{}-{}_saved_tracks.pickle'.format(seqs[seq]['date'], seqs[seq]['drive'], seqs[seq]['frames'][0], seqs[seq]['frames'][-1]))
    tm_fusion = run_fusion(kitti_basedir, seqs[seq]['date'], seqs[seq]['drive'], seqs[seq]['frames'], saved_tracks_filename, corrected_metrics_file)
    tm_baseline = TrajectoryMetrics.loadmat(orig_metrics_file)
    
    # Compute errors
    trans_armse_fusion, rot_armse_fusion = tm_fusion.mean_err()
    trans_armse_baseline, rot_armse_baseline = tm_baseline.mean_err()
    
    print('VO Only ARMSE (Trans / Rot): {:.3f} (m) / {:.3f} (a-a)'.format(trans_armse_baseline, rot_armse_baseline))
    print('SO(3) Fusion ARMSE (Trans / Rot): {:.3f} (m) / {:.3f} (a-a)'.format(trans_armse_fusion, rot_armse_fusion))

   



if __name__ == '__main__':
    # import cProfile, pstats
    # cProfile.run("run_svo()", "{}.profile".format(__file__))
    # s = pstats.Stats("{}.profile".format(__file__))
    # s.strip_dirs()
    # s.sort_stats("cumtime").print_stats(25)
    np.random.seed(14)
    main()

