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
    train_dataset_path = '../simulation/orbital/train_abs.mat'
    valid_dataset_path = '../simulation/orbital/valid_abs_ood.mat'

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

def create_sim_error_plot():
    valid_dataset = sio.loadmat(valid_dataset_path)
    num_hydra_heads=25
    model = QuaternionNet(D_in_sensor=valid_dataset_path['y_k_j'].shape[0]*valid_dataset_path['y_k_j'].shape[2], num_hydra_heads=num_hydra_heads)
    pretrained_model = torch.load('simulation/saved_plots/best_model_heads_1_epoch_74.pt')
    model.load_state_dict(pretrained_model['model'])
    avg_valid_loss, valid_ang_error, valid_nll, predict_history = validate(model, valid_loader, loss_fn, config, output_history=True)
    loss_fn = QuatNLLLoss()


def main():
    create_sim_world_plot()

if __name__ == '__main__':
    main()