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
    font_size = 24
    train_dataset_path = 'sim_data/train_abs.mat'
    valid_dataset_path = 'sim_data/valid_abs_ood.mat'

    train_dataset = sio.loadmat(train_dataset_path)
    valid_dataset = sio.loadmat(valid_dataset_path)

    T_vi_list = [SE3.from_matrix(train_dataset['T_vk_i'][i]) for i in range(train_dataset['T_vk_i'].shape[0])]
    T_vi_list_test = [SE3.from_matrix(valid_dataset['T_vk_i'][i]) for i in range(valid_dataset['T_vk_i'].shape[0])]

    # Output figure of the world
    radius = 25
    fig = plt.figure(figsize=(8, 5))
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
    ax.plot_surface(x, y, z, cmap=plt.cm.coolwarm, alpha=0.25)
    ax.scatter(r_vi_i[::5, 0], r_vi_i[::5, 1], r_vi_i[::5, 2], marker='<', s=4, color='firebrick', alpha=1, label='Training Poses')
    ax.scatter(r_vi_i_test[:, 0], r_vi_i_test[:, 1], r_vi_i_test[:, 2], marker='v', s=10, alpha=1, color='royalblue', label='Test Poses')
    ax.legend(fontsize=font_size, loc='upper left')
    ax.set_xlabel('x', fontsize=font_size)
    ax.set_ylabel('y', fontsize=font_size)
    ax.set_zlabel('z', fontsize=font_size)
    ax.xaxis.set_tick_params(labelsize=font_size - 10)
    ax.yaxis.set_tick_params(labelsize=font_size - 10)
    ax.zaxis.set_tick_params(labelsize=font_size - 10)
    fig.savefig('sim_world.png', bbox_inches='tight', dpi=300)

    return



def _plot_sigma(x, y, y_mean, y_sigma, y_sigma_2, label, ax, font_size=18):
    ax.fill_between(x, y_mean-3*y_sigma, y_mean+3*y_sigma, alpha=0.5, label='$\pm 3\sigma$ ($C$)', color='dodgerblue')
    ax.fill_between(x, y_mean - 3 * y_sigma_2, y_mean + 3 * y_sigma_2, alpha=0.5, color='red', label='$\pm 3\sigma$ ($\Sigma$ only)')
    ax.scatter(x, y, s=1, c='black')
    ax.set_ylabel(label, fontsize=font_size)
    return

def create_sim_error_plot():

    check_point = torch.load('sim_data/best_model_heads_25_epoch_90.pt')
    (q_gt, q_est, R_est, R_direct_est) = (check_point['predict_history'][0],
                                          check_point['predict_history'][1],
                                          check_point['predict_history'][2],
                                          check_point['predict_history'][3])

    fig, ax = plt.subplots(3, 1, sharex='col', sharey='row', figsize=(8, 5))

    x_labels =np.linspace(-81, 81, q_gt.shape[0])
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
    ax[2].legend(fontsize=font_size, loc='center')
    #image_array = canvas_to_array(fig)
    ax[2].xaxis.set_tick_params(labelsize=font_size-2)
    ax[0].yaxis.set_tick_params(labelsize=font_size-2)
    ax[1].yaxis.set_tick_params(labelsize=font_size-2)
    ax[2].yaxis.set_tick_params(labelsize=font_size-2)
    ax[2].set_xlabel('Semisphere angle ($\deg$)', fontsize=font_size)


    fig.savefig('sim_errors.png', bbox_inches='tight', dpi=300)


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

    fig_name = scene_checkpoint.split('/')[1].split('.')[0] + '.png'
    fig.savefig(fig_name, bbox_inches='tight', dpi=300)


def main():
    #create_sim_world_plot()
    #create_sim_error_plot()

    sevenscene_checkpoints = ['7scenes_data/best_model_chess_heads_25_epoch_5.pt',
                              '7scenes_data/best_model_pumpkin_heads_25_epoch_4.pt',
                              '7scenes_data/best_model_redkitchen_heads_25_epoch_4.pt']
    for checkpoint in sevenscene_checkpoints:
        create_7scenes_error_plot(checkpoint)

if __name__ == '__main__':
    main()