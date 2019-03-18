import pykitti
from matplotlib import pyplot as plt
import numpy as np
import torch
from liegroups import SE3, SO3
from pyslam.utils import invsqrt
from pyslam.metrics import TrajectoryMetrics
from optparse import OptionParser
from fusion_pipeline import SO3FusionPipeline
from pyslam.visualizers import TrajectoryVisualizer

import copy
import time
import os, glob
import pickle
import csv

import argparse


def _plot_sigma(x, err, sigma, label, ax):
    ax.fill_between(x, -3*sigma, 3*sigma, color='b', alpha=0.5, label='$\pm 3\sigma$ Total')
    ax.scatter(x, err, s=0.5, c='black')
    ax.set_ylabel(label)
    return


def plot_errors_with_sigmas(T_w_c_vo, T_w_c_gt, Sigma_est, filename='sigma_plot.pdf'):
    fig, ax = plt.subplots(6, 1, sharex='col', sharey='row')
    num_odom = len(T_w_c_gt) - 1

    x_labels = np.arange(0, num_odom)
    xi_errs = np.empty((num_odom, 6))
    nll = 0
    new_Sigma_est = copy.deepcopy(Sigma_est)

    for i in range(num_odom):

        T_21_est = T_w_c_vo[i+1].inv().dot(T_w_c_vo[i])
        T_21_gt = T_w_c_gt[i+1].inv().dot(T_w_c_gt[i])

        xi_errs_i = T_21_est.dot(T_21_gt.inv()).log()
        xi_errs[i] = xi_errs_i

        Sigma_i = np.diag(4*xi_errs_i**2)#0.5*Sigma_est[i, :, :]
        new_Sigma_est[i] = Sigma_i

        nll_i = 0.5*xi_errs_i.reshape(1,6).dot(np.linalg.inv(Sigma_i)).dot(xi_errs_i.reshape(6,1)) + 0.5*np.log(np.linalg.det(Sigma_i))
        nll += np.asscalar(nll_i)

    for i in range(6):
        _plot_sigma(x_labels, xi_errs[:, i], np.sqrt(new_Sigma_est[:,i,i].flatten()), '$\Xi_{}$ err'.format(i+1), ax[i])

    ax[0].set_title('NLL: {:.3f}'.format(nll/num_odom))
    fig.savefig(filename, bbox_inches='tight')
    plt.close(fig)



def plot_errors(baseline_metrics_file, seq):

    tm_vo = TrajectoryMetrics.loadmat(baseline_metrics_file)
    T_w_c_vo = tm_vo.Twv_est
    T_w_c_gt = tm_vo.Twv_gt
    Sigma_21_vo = tm_vo.mdict['Sigma_21']

    plot_errors_with_sigmas(T_w_c_gt, T_w_c_vo, Sigma_21_vo, filename='{}_vo_errs.pdf'.format(seq))




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


    seq = '02'
    tm_path = '../svo/baseline_tm/'

    hydranet_output_file = '../fusion/hydranet_output_reverse_model_seq_{}.pt'.format(seq)

    orig_metrics_file = os.path.join(tm_path, '{}_drive_{}.mat'.format(seqs[seq]['date'],seqs[seq]['drive']))

    plot_errors(orig_metrics_file, seq)




if __name__ == '__main__':
    # import cProfile, pstats
    # cProfile.run("run_svo()", "{}.profile".format(__file__))
    # s = pstats.Stats("{}.profile".format(__file__))
    # s.strip_dirs()
    # s.sort_stats("cumtime").print_stats(25)
    np.random.seed(14)
    main()

