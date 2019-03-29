import numpy as np
import torch
from torch import autograd
import os
import scipy.io as sio
import sys
sys.path.insert(0,'..')
import matplotlib
matplotlib.use('Agg')
from liegroups.numpy import SE3, SO3
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.io as sio
from utils import *
from pyslam.metrics import TrajectoryMetrics
from pyslam.visualizers import TrajectoryVisualizer
import csv


plt.rc('text', usetex=True)
plt.rc('font', family='serif')


def _plot_sigma_with_gt(x, y_est, y_gt, y_sigma, label, ax, y_lim=None):

    #ax.fill_between(x, y_est-y_sigma, y_est+y_sigma, alpha=0.2, facecolor='dodgerblue', label='$\pm \sigma$')
    #ax.fill_between(x, y_est-2*y_sigma, y_est+2*y_sigma, alpha=0.5, facecolor='dodgerblue', label='$\pm 2\sigma$')
    ax.fill_between(x, y_est-3*y_sigma, y_est+3*y_sigma, alpha=0.9, facecolor='dodgerblue', label='$\pm 3\sigma$', rasterized=False)
    ax.plot(x, y_gt,  c='black', linewidth=0.75, label='GT',rasterized=False)
    ax.plot(x, y_est, c='green', linewidth=0.75, label='HydraNet', rasterized=False)
    ax.set_ylabel(label)
    if y_lim is not None:
        ax.set_ylim(y_lim)
    return

def _plot_hist(x, ax, axis):
    colours = ['g', 'r', 'b']
    ax.hist(x, 100, [-0.5,0.5], density=False, facecolor=colours[axis], alpha=0.25, label='$\Delta \phi_{}$'.format(axis), rasterized=True)

def create_kitti_histogram(seqs):
    fig, ax = plt.subplots(1, 3, figsize=(6, 2))
    for s_i, seq in enumerate(seqs):
        hn_path = '../kitti/fusion/hydranet_output_reverse_model_seq_{}.pt'.format(seq)
        hn_data = torch.load(hn_path)
        C_21_hn_est = hn_data['Rot_21'].numpy()
        C_21_hn_gt = hn_data['Rot_21_gt'].numpy()
        Sigma_21 = hn_data['Sigma_21'].numpy()

        num_odom = len(C_21_hn_gt)
        phi_errs = np.empty((num_odom, 3))
        for pose_i in range(num_odom):
            C_21_est = SO3.from_matrix(C_21_hn_est[pose_i], normalize=True)
            C_21_gt = SO3.from_matrix(C_21_hn_gt[pose_i], normalize=True)
            phi_errs_i = C_21_est.dot(C_21_gt.inv()).log()
            phi_errs[pose_i] = phi_errs_i*180./np.pi

        ax[s_i].grid()
        ax[s_i].set_title(seq)
        ax[s_i].set_xlabel('(deg)')
        ax[s_i].set_ylim([0, 600])
        if s_i > 0:
            ax[s_i].get_yaxis().set_ticklabels([])
        for i in range(3):
            _plot_hist(phi_errs[:,i], ax[s_i], i)

    ax[2].legend()
    fig.savefig('kitti_so3_hist.pdf', bbox_inches='tight')
    plt.close(fig)

def compute_kitti_nll_and_error(seqs):
    stats_list = []
    for s_i, seq in enumerate(seqs):
        hn_path = '../kitti/fusion/hydranet_output_reverse_model_seq_{}.pt'.format(seq)
        hn_data = torch.load(hn_path)
    
        C_21_hn_est = hn_data['Rot_21']
        C_21_hn_gt = hn_data['Rot_21_gt']
        Sigma_21 = hn_data['Sigma_21']
        
        nll = nll_mat(C_21_hn_est, C_21_hn_gt, Sigma_21.inverse())
        mean_nll = nll.mean().numpy()
    
    
        C_21_hn_est = hn_data['Rot_21'].numpy()
        C_21_hn_gt = hn_data['Rot_21_gt'].numpy()
        Sigma_21 = hn_data['Sigma_21'].numpy()
        num_odom = len(C_21_hn_gt)
        phi_errs = np.empty((num_odom, 3))
        norm_errs = np.empty((num_odom,1))
        for pose_i in range(num_odom):
            C_21_est = SO3.from_matrix(C_21_hn_est[pose_i], normalize=True)
            C_21_gt = SO3.from_matrix(C_21_hn_gt[pose_i], normalize=True)
            phi_errs_i = C_21_est.dot(C_21_gt.inv()).log()
            phi_errs[pose_i] = phi_errs_i*180./np.pi
            norm_errs[pose_i] = np.linalg.norm(phi_errs[pose_i])
        axis_errors = np.abs(phi_errs).mean(axis=0)
        mean_error = norm_errs.mean()
        
        for i in range(0, axis_errors.shape[0]):
            stats_list.append([seq, i, axis_errors[i], mean_nll])
        stats_list.append([seq, i+1, mean_error, ''])
    
    csv_filename = 'kitti_nll_error_stats.csv'
    csv_header = ['Sequence', 'axis','Error', 'NLL']
    with open(csv_filename, "w") as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)
        writer.writerows(stats_list)
    
def create_kitti_abs_with_sigmas_plot(seq):
    hn_path = '../kitti/fusion/hydranet_output_reverse_model_seq_{}.pt'.format(seq)
    hn_data = torch.load(hn_path)

    C_21_hn_est = hn_data['Rot_21'].numpy()
    C_21_hn_gt = hn_data['Rot_21_gt'].numpy()
    Sigma_21 = hn_data['Sigma_21'].numpy()

    fig, ax = plt.subplots(3, 1, sharex='col', sharey='row')

    x_labels = np.arange(0, C_21_hn_est.shape[0])

    num_odom = C_21_hn_est.shape[0]
    phi_est = np.empty((num_odom, 3))
    phi_gt = np.empty((num_odom, 3))

    deg_factor = 180./np.pi

    for pose_i in range(num_odom):
        phi_est[pose_i] = SO3.from_matrix(C_21_hn_est[pose_i], normalize=True).log() * deg_factor
        phi_gt[pose_i] = SO3.from_matrix(C_21_hn_gt[pose_i], normalize=True).log() * deg_factor


    _plot_sigma_with_gt(x_labels, phi_est[:, 0], phi_gt[:, 0], np.sqrt(Sigma_21[:,0,0].flatten()) * deg_factor, '$\phi_1$ (deg)', ax[0])
    _plot_sigma_with_gt(x_labels, phi_est[:, 1], phi_gt[:, 1], np.sqrt(Sigma_21[:,1,1].flatten()) * deg_factor, '$\phi_2$ (deg)', ax[1])
    _plot_sigma_with_gt(x_labels, phi_est[:, 2], phi_gt[:, 2], np.sqrt(Sigma_21[:,2,2].flatten()) * deg_factor, '$\phi_3$ (deg)', ax[2])

    ax[2].legend()
    #image_array = canvas_to_array(fig)
    output_file = 'kitti_abs_{}.pdf'.format(seq)
    fig.savefig(output_file, bbox_inches='tight')
    plt.close(fig)

def create_kitti_topdown_plot(tm_svo, tm_fusion, seq):
    tm_dict = {'viso2-s': tm_svo, 'Fusion': tm_fusion}
    vis = TrajectoryVisualizer(tm_dict)
    fig, ax = vis.plot_topdown(which_plane='xy')
    output_file = 'kitti_{}_topdown.pdf'.format(seq)
    fig.savefig(output_file, bbox_inches='tight')


def create_kitti_seg_err_plots(tm_svo, tm_fusion, seq, segs=list(range(100, 801, 100))):
    tm_dict = {'viso2-s': tm_svo, 'Fusion': tm_fusion}
    vis = TrajectoryVisualizer(tm_dict)
    fig, ax = vis.plot_segment_errors(segs)
    output_file = 'kitti_{}_seg_err.pdf'.format(seq)
    fig.savefig(output_file)



def output_kitti_stats(tm_svo, tm_fusion, seq, segs=list(range(100, 801, 100))):


    csv_header = ['Sequence', 'Type', 'Traj t','Traj r', 'Rel t (delta=1)', 'Rel r (delta=1)', 'Rel t (delta=5)', 'Rel r (delta=5)', 'Seg t', 'Seg r']

    trans_armse_fusion_traj, rot_armse_fusion_traj = tm_fusion.mean_err(error_type='traj', rot_unit='deg')
    trans_armse_vo_traj, rot_armse_vo_traj = tm_svo.mean_err(error_type='traj', rot_unit='deg')

    trans_rel_1_fusion, rot_rel_1_fusion = tm_fusion.error_norms(error_type='rel', rot_unit='deg', delta=1)
    trans_rel_1_vo, rot_rel_1_vo = tm_svo.error_norms(error_type='rel', rot_unit='deg', delta=1)

    trans_rel_1_fusion, rot_rel_1_fusion = np.mean(trans_rel_1_fusion),  np.mean(rot_rel_1_fusion)
    trans_rel_1_vo, rot_rel_1_vo = np.mean(trans_rel_1_vo), np.mean(rot_rel_1_vo)

    trans_rel_5_fusion, rot_rel_5_fusion = tm_fusion.error_norms(error_type='rel', rot_unit='deg', delta=5)
    trans_rel_5_vo, rot_rel_5_vo = tm_svo.error_norms(error_type='rel', rot_unit='deg', delta=5)

    trans_rel_5_fusion, rot_rel_5_fusion = np.mean(trans_rel_5_fusion), np.mean(rot_rel_5_fusion)
    trans_rel_5_vo, rot_rel_5_vo = np.mean(trans_rel_5_vo), np.mean(rot_rel_5_vo)

    _, fusion_avg_errs = tm_fusion.segment_errors(segs, rot_unit='deg')
    _, vo_avg_errs = tm_svo.segment_errors(segs, rot_unit='deg')

    fus_seg_errs = [np.mean(fusion_avg_errs[:,1]), np.mean(fusion_avg_errs[:,2])]
    vo_seg_errs = [np.mean(vo_avg_errs[:,1]), np.mean(vo_avg_errs[:,2])]

    vo_stats =  [seq, 'viso2-s',
                   trans_armse_vo_traj, rot_armse_vo_traj,
                   trans_rel_1_vo, rot_rel_1_vo,
                   trans_rel_5_vo, rot_rel_5_vo,
                   vo_seg_errs[0], vo_seg_errs[1]]

    fusion_stats = [seq, 'fusion',
                  trans_armse_fusion_traj, rot_armse_fusion_traj,
                  trans_rel_1_fusion, rot_rel_1_fusion,
                  trans_rel_5_fusion, rot_rel_5_fusion,
                  fus_seg_errs[0], fus_seg_errs[1]]

    csv_filename = '{}_stats.csv'.format(seq)
    with open(csv_filename, "w") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
        writer.writerow(csv_header)
        writer.writerow(vo_stats)
        writer.writerow(fusion_stats)


def main():

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

    process_seqs = ['00', '02', '05']
    segs = list(range(100, 801, 100))

    for seq in process_seqs:
        svo_tm_path = '../kitti/svo/baseline_tm/'
        fusion_tm_output_path = '../kitti/fusion/fusion_tms/'

        fusion_file = os.path.join(fusion_tm_output_path,
                                           'SO3_fused_single_{}_drive_{}.mat'.format(seqs[seq]['date'],
                                                                                     seqs[seq]['drive']))

        svo_file = os.path.join(svo_tm_path, '{}_drive_{}.mat'.format(seqs[seq]['date'], seqs[seq]['drive']))

        # tm_svo = TrajectoryMetrics.loadmat(svo_file)
        # tm_fusion = TrajectoryMetrics.loadmat(fusion_file)
        #create_kitti_topdown_plots(tm_svo, tm_fusion, seq)
        #output_kitti_stats(tm_svo, tm_fusion, seq, segs)
        #create_kitti_seg_err_plots(tm_svo, tm_fusion, seq, segs)
#        create_kitti_abs_with_sigmas_plot(seq)
        

    #create_kitti_histogram(process_seqs)
    compute_kitti_nll_and_error(process_seqs)

if __name__ == '__main__':
    main()