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

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# pretrained_model = torch.load('simulation/saved_plots/best_model_heads_1_epoch_74.pt')
# model.sensor_net.load_state_dict(pretrained_model['sensor_net'])
# model.direct_covar_head.load_state_dict(pretrained_model['direct_covar_head'])

def create_semisphere(radius, num_pts):
    #Create a semi-sphere of points 'evenly' spaced on a semi-sphere (evenly in terms of angular metrics)
    #Outputs 'grids' that can be used with matplotlib's plot_surface

    #Th: Polar
    #Phi: Azimuth
    #For a semi-sphere: th: 0-90, phi: 0-360
    th = np.linspace(0., np.pi/2., np.sqrt(num_pts))
    phi = np.linspace(0., 2.*np.pi, np.sqrt(num_pts))
    th_grid, phi_grid = np.meshgrid(th, phi)

    x = radius*np.sin(th_grid)*np.cos(phi_grid)
    y = radius*np.sin(th_grid)*np.sin(phi_grid)
    z = radius*np.cos(th_grid)

    return (x,y,z)

def create_sim_world_plot():
    font_size = 12
    train_dataset_path = 'sim_data/train_abs.mat'
    valid_dataset_path = 'sim_data/valid_abs_ood.mat'

    train_dataset = sio.loadmat(train_dataset_path)
    valid_dataset = sio.loadmat(valid_dataset_path)

    T_vi_list = [SE3.from_matrix(train_dataset['T_vk_i'][i]) for i in range(train_dataset['T_vk_i'].shape[0])]
    T_vi_list_test = [SE3.from_matrix(valid_dataset['T_vk_i'][i]) for i in range(valid_dataset['T_vk_i'].shape[0])]

    # Output figure of the world
    radius = 25
    fig = plt.figure(figsize=(6, 4))
    ax = Axes3D(fig)
    r_vi_i = np.empty((len(T_vi_list), 3))
    for i in range(len(T_vi_list)):
        r_vi_i[i] = T_vi_list[i].inv().trans

    r_vi_i_test = np.empty((len(T_vi_list_test), 3))
    for i in range(len(T_vi_list_test)):
        r_vi_i_test[i] = T_vi_list_test[i].inv().trans


    (x, y, z) = create_semisphere(radius, 400)
    pts = train_dataset['rho_i_pj_i'].T
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], label='Landmarks', s=10., color='limegreen')
    ax.plot_surface(x, y, z, cmap=plt.cm.coolwarm, alpha=0.25, rasterized=True)
    ax.scatter(r_vi_i[::5, 0], r_vi_i[::5, 1], r_vi_i[::5, 2], marker='<', s=4, color='firebrick', alpha=1, label='Training Poses', rasterized=True)
    ax.scatter(r_vi_i_test[:, 0], r_vi_i_test[:, 1], r_vi_i_test[:, 2], marker='v', s=10, alpha=1, color='royalblue', label='Test Poses', rasterized=True)
    ax.legend(fontsize=font_size, loc='upper left')
    ax.set_xlabel('x', fontsize=font_size)
    ax.set_ylabel('y', fontsize=font_size)
    ax.set_zlabel('z', fontsize=font_size)
    ax.xaxis.set_tick_params(labelsize=font_size)
    ax.yaxis.set_tick_params(labelsize=font_size)
    ax.zaxis.set_tick_params(labelsize=font_size)
    fig.savefig('sim_world.pdf', bbox_inches='tight', dpi=300)

    return



def _plot_sigma(x, y, y_mean, y_sigma, y_sigma_2, label, ax, font_size=18):
    ax.fill_between(x, y_mean-3*y_sigma, y_mean+3*y_sigma, alpha=0.5, label='$\pm 3\sigma$ ($\Sigma_t$)', color='dodgerblue')
    ax.fill_between(x, y_mean - 3 * y_sigma_2, y_mean + 3 * y_sigma_2, alpha=0.5, color='red', label='$\pm 3\sigma$ ($\Sigma_d$)')
    ax.scatter(x, y, s=1, c='black', label='Rotational Error')
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

def create_sim_error_plot():

    check_point = torch.load('sim_data/best_model_heads_25_epoch_90.pt')
    (q_gt, q_est, R_est, R_direct_est) = (check_point['predict_history'][0],
                                          check_point['predict_history'][1],
                                          check_point['predict_history'][2],
                                          check_point['predict_history'][3])

    fig, ax = plt.subplots(3, 1, sharex='col', sharey='row', figsize=(6, 4))

    x_labels =np.linspace(-81, 81, q_gt.shape[0])
    phi_errs = quat_log_diff(q_est, q_gt).numpy()
    R_est = R_est.numpy()
    R_direct_est = R_direct_est.numpy()
    font_size = 10


    _plot_sigma(x_labels, phi_errs[:, 0], 0., np.sqrt(R_est[:, 0, 0].flatten()),
                np.sqrt(R_direct_est[:, 0, 0].flatten()), '$\phi_1$ err', ax[0], font_size=font_size)
    _plot_sigma(x_labels, phi_errs[:, 1], 0., np.sqrt(R_est[:, 1, 1].flatten()),
                np.sqrt(R_direct_est[:, 1, 1].flatten()), '$\phi_2$ err', ax[1], font_size=font_size)
    _plot_sigma(x_labels, phi_errs[:, 2], 0., np.sqrt(R_est[:, 2, 2].flatten()),
                np.sqrt(R_direct_est[:, 2, 2].flatten()), '$\phi_3$ err', ax[2], font_size=font_size)
    ax[2].legend(fontsize=font_size, loc='center')
    #image_array = canvas_to_array(fig)
    ax[2].xaxis.set_tick_params(labelsize=font_size-2)
    ax[0].yaxis.set_tick_params(labelsize=font_size-2)
    ax[1].yaxis.set_tick_params(labelsize=font_size-2)
    ax[2].yaxis.set_tick_params(labelsize=font_size-2)
    ax[2].set_xlabel('Polar Angle ($\circ$)', fontsize=font_size)
    fig.savefig('sim_errors.pdf', bbox_inches='tight', dpi=300)


def create_7scenes_error_plot(scene_checkpoint):
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


def create_7scenes_histogram_plot(scene_checkpoint):
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
    
def main():
    #create_sim_world_plot()
    create_sim_error_plot()

    sevenscene_checkpoints = ['7scenes_data/best_model_chess_heads_25_epoch_13.pt',
                              '7scenes_data/best_model_fire_heads_25_epoch_4.pt',
                              '7scenes_data/best_model_heads_heads_25_epoch_11.pt']
    for checkpoint in sevenscene_checkpoints:
        create_7scenes_error_plot(checkpoint)
        create_7scenes_histogram_plot(checkpoint)
        create_7scenes_abs_with_sigmas_plot(checkpoint)


if __name__ == '__main__':
    main()