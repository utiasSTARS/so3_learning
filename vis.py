import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from utils import *
import io
from PIL import Image
from liegroups.torch import SE3

def canvas_to_array(fig):    
    #Convert matplotlib figure to a C X H X W numpy array for TensorboardX
    canvas = fig.canvas
    canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi()
    image_np = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(height.astype(np.uint32), width.astype(np.uint32), 3)
    #PIL expects CXHXW
    return np.rollaxis(image_np, 2)



def _plot_sigma(x, y, y_mean, y_sigma, y_sigma_2, label, ax):
    ax.scatter(x, y, s=0.5, c='black')
    ax.fill_between(x, y_mean-3*y_sigma, y_mean+3*y_sigma, alpha=0.5, label='$\pm 3\sigma$ Total')
    ax.fill_between(x, y_mean - 3 * y_sigma_2, y_mean + 3 * y_sigma_2, alpha=0.5, color='red', label='$\pm 3\sigma$ Direct')
    ax.set_ylabel(label)
    return

def _plot_sigma_with_gt(x, y_est, y_gt, y_sigma, y_sigma_2, label, ax):
    ax.scatter(x, y_est, s=0.5, c='green')
    ax.scatter(x, y_gt, s=0.5, c='black')
    ax.fill_between(x, y_est-3*y_sigma, y_est+3*y_sigma, alpha=0.5, label='$\pm 3\sigma$ Total')
    ax.fill_between(x, y_est - 3 * y_sigma_2, y_est + 3 * y_sigma_2, alpha=0.5, color='red', label='$\pm 3\sigma$ Direct')
    ax.set_ylabel(label)
    return
def plot_errors_with_sigmas(q_gt, q_est, R_est, R_direct_est, filename='sigma_plot.pdf'):
    fig, ax = plt.subplots(3, 1, sharex='col', sharey='row')

    x_labels = np.arange(0, q_gt.shape[0])
    phi_errs = quat_log_diff(q_est, q_gt).numpy()
    R_est = R_est.numpy()
    R_direct_est = R_direct_est.numpy()

    _plot_sigma(x_labels, phi_errs[:, 0], 0., np.sqrt(R_est[:,0,0].flatten()), np.sqrt(R_direct_est[:,0,0].flatten()),  '$\Theta_1$ err', ax[0])
    _plot_sigma(x_labels, phi_errs[:, 1], 0., np.sqrt(R_est[:,1,1].flatten()), np.sqrt(R_direct_est[:,1,1].flatten()), '$\Theta_2$ err', ax[1])
    _plot_sigma(x_labels, phi_errs[:, 2], 0., np.sqrt(R_est[:,2,2].flatten()), np.sqrt(R_direct_est[:,2,2].flatten()), '$\Theta_3$ err', ax[2])

    ax[2].legend()
    #image_array = canvas_to_array(fig)
    fig.savefig(filename, bbox_inches='tight')
    plt.close(fig)

def plot_abs_with_sigmas(q_gt, q_est, R_est, R_direct_est, filename='sigma_plot.pdf'):
    fig, ax = plt.subplots(3, 1, sharex='col', sharey='row')

    x_labels = np.arange(0, q_gt.shape[0])
    phi_est = quat_log(q_est).numpy()
    phi_gt = quat_log(q_gt).numpy()

    R_est = R_est.numpy()
    R_direct_est = R_direct_est.numpy()

    _plot_sigma_with_gt(x_labels, phi_gt[:, 0], phi_est[:, 0], np.sqrt(R_est[:,0,0].flatten()), np.sqrt(R_direct_est[:,0,0].flatten()),  '$\Theta_1$ err', ax[0])
    _plot_sigma_with_gt(x_labels, phi_gt[:, 1], phi_est[:, 1], np.sqrt(R_est[:,1,1].flatten()), np.sqrt(R_direct_est[:,1,1].flatten()), '$\Theta_2$ err', ax[1])
    _plot_sigma_with_gt(x_labels, phi_gt[:, 2], phi_est[:, 2], np.sqrt(R_est[:,2,2].flatten()), np.sqrt(R_direct_est[:,2,2].flatten()), '$\Theta_3$ err', ax[2])

    ax[2].legend()
    #image_array = canvas_to_array(fig)
    fig.savefig(filename, bbox_inches='tight')
    plt.close(fig)

def plot_nees(q_gt, q_est, R_est, filename='nees_plot.pdf'):
    fig, ax = plt.subplots(1, 1, sharex='col', sharey='row')
    xlabels = np.arange(0, q_gt.shape[0])
    residuals = quat_log_diff(q_est, q_gt).unsqueeze(2)
    nees = (1./3.) * residuals.transpose(1, 2).bmm(R_est.inverse()).bmm(residuals)
    nees.squeeze_()

    ax.plot(xlabels, nees.numpy(), label='nees')
    ax.legend()
    #image_array = canvas_to_array(fig)
    fig.savefig(filename, bbox_inches='tight')
    plt.close(fig)

def plot_3D(tau_gt, tau_odom, tau_est, l_true=None):
    fig = plt.figure()
    ax = Axes3D(fig)
    # ax.set_zlim([0, 2.5])
    # ax.set_ylim([0, 5])
    # ax.set_xlim([1, 3.5])
    if l_true is not None:
        ax.scatter(l_true[0,:], l_true[1,:], l_true[2,:], label='Landmarks')


    r_odom = SE3.exp(torch.from_numpy(tau_odom)).inv().trans.numpy()
    r_est = SE3.exp(torch.from_numpy(tau_est)).inv().trans.numpy()
    r_gt = SE3.exp(torch.from_numpy(tau_gt)).inv().trans.numpy()
    
    ax.plot3D(r_odom[:,0], r_odom[:, 1],r_odom[:, 2], '-r', label='IMU Integration')
    ax.plot3D(r_est[:,0], r_est[:, 1], r_est[:, 2], '-g', label='Corrected')
    ax.plot3D(r_gt[:,0], r_gt[:, 1], r_gt[:, 2], '--k', label='Ground Truth')
    ax.legend()
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.view_init(elev=90., azim=0.)
    image_array = canvas_to_array(fig)
    plt.close(fig)
    return image_array