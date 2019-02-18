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



def plot_norms(t, tau_gt, tau_odom, tau_est):
    fig, ax = plt.subplots(2, 1, sharex='col', sharey='row')
    
    (odom_r_norms, odom_frob_norms) = compute_norms(tau_gt, tau_odom)
    (est_r_norms, est_frob_norms) = compute_norms(tau_gt, tau_est)

    ax[0].plot(t.flatten(), odom_r_norms, '--', label='Odom')
    ax[0].plot(t.flatten(), est_r_norms, label='Corr')
    #plt.grid()
    ax[0].set_title('Translation Norms')
    ax[0].legend()
    
    ax[1].plot(t, odom_frob_norms, '--', label='Odom')
    ax[1].plot(t, est_frob_norms, label='Corr')
    #plt.grid()
    ax[1].set_title('Rotation Norms')
    ax[1].legend()
    
    image_array = canvas_to_array(fig)
    plt.close(fig)
    return image_array

def _plot_sigma(x, y, y_mean, y_sigma, label, ax):
    ax.plot(x, y, label=label)
    ax.fill_between(x, y_mean-3*y_sigma, y_mean+3*y_sigma, alpha=0.25, label='$\pm 3\sigma$')
    ax.legend()
    return

def plot_errors_with_sigmas(t, tau_gt, tau_est, P_est):
    fig, ax = plt.subplots(6, 1, sharex='col', sharey='row')
    
    tau_errs = se3_errs(tau_gt, tau_est)
    r_errs = tau_errs[:, :3] 
    phi_errs = tau_errs[:, 3:]

    _plot_sigma(t.flatten(), r_errs[:, 0], 0., np.sqrt(P_est[:,0,0].flatten()), 'x err', ax[0])
    _plot_sigma(t.flatten(), r_errs[:, 1], 0., np.sqrt(P_est[:,1,1].flatten()), 'y err', ax[1])
    _plot_sigma(t.flatten(), r_errs[:, 2], 0., np.sqrt(P_est[:,2,2].flatten()), 'z err', ax[2])

    _plot_sigma(t.flatten(), phi_errs[:, 0], 0., np.sqrt(P_est[:,3,3].flatten()), '$\Theta_1$ err', ax[3])
    _plot_sigma(t.flatten(), phi_errs[:, 1], 0., np.sqrt(P_est[:,4,4].flatten()), '$\Theta_2$ err', ax[4])
    _plot_sigma(t.flatten(), phi_errs[:, 2], 0., np.sqrt(P_est[:,5,5].flatten()), '$\Theta_3$ err', ax[5])

    image_array = canvas_to_array(fig)
    plt.close(fig)
    return image_array

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