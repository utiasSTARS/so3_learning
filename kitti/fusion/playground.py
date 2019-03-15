import pykitti
from matplotlib import pyplot as plt
import numpy as np
import torch
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

    seq = '04'
    tm_path = '../svo/baseline_tm/'

    orig_metrics_file = os.path.join(tm_path, '{}_drive_{}.mat'.format(seqs[seq]['date'], seqs[seq]['drive']))
    hydranet_output_file = 'hydranet_output_model_seq_{}.pt'.format(seq)

    tm_baseline = TrajectoryMetrics.loadmat(orig_metrics_file)

    # Compute errors
    trans_armse_baseline, rot_armse_baseline = tm_baseline.mean_err(error_type='rel', rot_unit='deg')

    print('VO Only ARMSE (Trans / Rot): {:.3f} (m) / {:.3f} (a-a)'.format(trans_armse_baseline, rot_armse_baseline))


if __name__ == '__main__':
    # import cProfile, pstats
    # cProfile.run("run_svo()", "{}.profile".format(__file__))
    # s = pstats.Stats("{}.profile".format(__file__))
    # s.strip_dirs()
    # s.sort_stats("cumtime").print_stats(25)
    np.random.seed(14)
    main()

