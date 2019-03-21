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

# pretrained_model = torch.load('simulation/saved_plots/best_model_heads_1_epoch_74.pt')
# model.sensor_net.load_state_dict(pretrained_model['sensor_net'])
# model.direct_covar_head.load_state_dict(pretrained_model['direct_covar_head'])

def _plot_sigma(x, y, y_mean, y_sigma, y_sigma_2, label, ax, font_size=18):
    ax.fill_between(x, y_mean-3*y_sigma, y_mean+3*y_sigma, alpha=0.5, label='$\pm 3\sigma$ ($C$)', color='dodgerblue')
    ax.fill_between(x, y_mean - 3 * y_sigma_2, y_mean + 3 * y_sigma_2, alpha=0.5, color='red', label='$\pm 3\sigma$ ($\Sigma$ only)')
    ax.scatter(x, y, s=1, c='black')
    ax.set_ylabel(label, fontsize=font_size)
    return

def _plot_hist(x, ax):
    for i in range(0,x.shape[1]):
        ax[i].hist(x[:,i], 100, density=True, facecolor='g', alpha=0.75)
        ax[i].grid()

def _plot_sigma_with_gt(x, y_est, y_gt, y_sigma, y_sigma_2, label, ax, y_lim=None):
    ax.fill_between(x, y_est-3*y_sigma, y_est+3*y_sigma, alpha=0.5, label='$\pm 3\sigma$ Total')
    ax.fill_between(x, y_est - 3 * y_sigma_2, y_est + 3 * y_sigma_2, alpha=0.5, color='red', label='$\pm 3\sigma$ Direct')
    ax.scatter(x, y_est, s=0.5, c='green')
    ax.scatter(x, y_gt, s=0.5, c='black')
    ax.set_ylabel(label)
    if y_lim is not None:
        ax.set_ylim(y_lim)
    return


def create_kitti_error_plot(scene_checkpoint):
    check_point = torch.load(scene_checkpoint, map_location=lambda storage, loc: storage)
    (q_gt, q_est, R_est, R_direct_est) = (check_point['predict_history'][0],
                                          check_point['predict_history'][1],
                                          check_point['predict_history'][2],
                                          check_point['predict_history'][3])

    fig, ax = plt.subplots(3, 1, sharex='col', sharey='row', figsize=(6, 8))

    x_labels =np.arange(0, q_gt.shape[0])
    phi_errs = quat_log_diff(q_est, q_gt).numpy()
    R_est = R_est.numpy()
    R_direct_est = R_direct_est.numpy()
    font_size = 18


    _plot_sigma(x_labels, phi_errs[:, 0], 0., np.sqrt(R_est[:, 0, 0].flatten()),
                np.sqrt(R_direct_est[:, 0, 0].flatten()), '$\phi_1$ err', ax[0], font_size=font_size)
    _plot_sigma(x_labels, phi_errs[:, 1], 0., np.sqrt(R_est[:, 1, 1].flatten()),
                np.sqrt(R_direct_est[:, 1, 1].flatten()), '$\phi_2$ err', ax[1], font_size=font_size)
    _plot_sigma(x_labels, phi_errs[:, 2], 0., np.sqrt(R_est[:, 2, 2].flatten()),
                np.sqrt(R_direct_est[:, 2, 2].flatten()), '$\phi_3$ err', ax[2], font_size=font_size)
    #ax[2].legend(fontsize=font_size, loc='center')
    #image_array = canvas_to_array(fig)
    ax[2].xaxis.set_tick_params(labelsize=font_size-2)
    ax[0].yaxis.set_tick_params(labelsize=font_size-2)
    ax[1].yaxis.set_tick_params(labelsize=font_size-2)
    ax[2].yaxis.set_tick_params(labelsize=font_size-2)
    ax[2].set_xlabel('Pose', fontsize=font_size)

#    fig_name = scene_checkpoint.split('/')[1].split('.')[0] + '.png'
    output_file = '7scenes_err_' + scene_checkpoint.replace('.pt','').replace('7scenes_data/','') + '.pdf'
    fig.savefig(output_file, bbox_inches='tight', dpi=300)


def create_kitti_histogram_plot(scene_checkpoint):
    check_point = torch.load(scene_checkpoint, map_location=lambda storage, loc: storage)
    (q_gt, q_est, R_est, R_direct_est) = (check_point['predict_history'][0],
                                          check_point['predict_history'][1],
                                          check_point['predict_history'][2],
                                          check_point['predict_history'][3])

    fig, ax = plt.subplots(3, 1, sharex='col', sharey='row', figsize=(6, 8))
    font_size = 18
#    x_labels =np.arange(0, q_gt.shape[0])
    phi_errs = quat_log_diff(q_est, q_gt).numpy()
    fig, ax = plt.subplots(3, 1, sharex='col', sharey='row')
    _plot_hist(phi_errs, ax)

    output_file = '7scenes_hist_' + scene_checkpoint.replace('.pt','').replace('7scenes_data/','') + '.pdf'
    ax[2].set_xlabel('Error (rad)', fontsize=font_size)
    fig.savefig(output_file, bbox_inches='tight')
    plt.close(fig)

def create_7scenes_abs_with_sigmas_plot(scene_checkpoint):
    check_point = torch.load(scene_checkpoint, map_location=lambda storage, loc: storage)
    (q_gt, q_est, R_est, R_direct_est) = (check_point['predict_history'][0],
                                          check_point['predict_history'][1],
                                          check_point['predict_history'][2],
                                          check_point['predict_history'][3])

    fig, ax = plt.subplots(3, 1, sharex='col', sharey='row')

    x_labels = np.arange(0, q_gt.shape[0])
    phi_est = quat_log(q_est).numpy()
    phi_gt = quat_log(q_gt).numpy()

    R_est = R_est.numpy()
    R_direct_est = R_direct_est.numpy()

    _plot_sigma_with_gt(x_labels, phi_est[:, 0], phi_gt[:, 0], np.sqrt(R_est[:,0,0].flatten()), np.sqrt(R_direct_est[:,0,0].flatten()),  '$\Theta_1$', ax[0])
    _plot_sigma_with_gt(x_labels, phi_est[:, 1], phi_gt[:, 1], np.sqrt(R_est[:,1,1].flatten()), np.sqrt(R_direct_est[:,1,1].flatten()), '$\Theta_2$', ax[1])
    _plot_sigma_with_gt(x_labels, phi_est[:, 2], phi_gt[:, 2], np.sqrt(R_est[:,2,2].flatten()), np.sqrt(R_direct_est[:,2,2].flatten()), '$\Theta_3$', ax[2])

    ax[2].legend()
    #image_array = canvas_to_array(fig)
    output_file = '7scenes_abs_' + scene_checkpoint.replace('.pt','').replace('7scenes_data/','') + '.pdf'
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

    process_seqs = {'00', '02', '05'}
    segs = list(range(100, 801, 100))

    for seq in process_seqs:
        svo_tm_path = '../kitti/svo/baseline_tm/'
        fusion_tm_output_path = '../kitti/fusion/fusion_tms/'

        fusion_file = os.path.join(fusion_tm_output_path,
                                           'SO3_fused_single_{}_drive_{}.mat'.format(seqs[seq]['date'],
                                                                                     seqs[seq]['drive']))

        svo_file = os.path.join(svo_tm_path, '{}_drive_{}.mat'.format(seqs[seq]['date'], seqs[seq]['drive']))

        tm_svo = TrajectoryMetrics.loadmat(svo_file)
        tm_fusion = TrajectoryMetrics.loadmat(fusion_file)
        #create_kitti_topdown_plots(tm_svo, tm_fusion, seq)
        #output_kitti_stats(tm_svo, tm_fusion, seq, segs)
        create_kitti_seg_err_plots(tm_svo, tm_fusion, seq, segs)
    #create_sim_world_plot()
    #create_sim_error_plot()



if __name__ == '__main__':
    main()