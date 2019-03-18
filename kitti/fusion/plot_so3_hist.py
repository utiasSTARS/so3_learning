from matplotlib import pyplot as plt
import numpy as np
import torch
from liegroups.numpy import SO3

def _plot_hist(x, ax):
    ax.hist(x, 100, density=True, facecolor='g', alpha=0.75)
    ax.grid()


def make_histogram(hn_path, filename):
    hn_data = torch.load(hn_path)
    C_21_hn_est = hn_data['Rot_21'].numpy()
    C_12_hn_est = hn_data['Rot_12'].numpy()
    C_21_hn_gt = hn_data['Rot_21_gt'].numpy()


    num_odom = len(C_21_hn_gt)
    phi_errs = np.empty((num_odom, 3))
    phi_errs_l = []
    for pose_i in range(num_odom):
        C_21_est = SO3.from_matrix(C_21_hn_est[pose_i], normalize=True)
        C_21_gt = SO3.from_matrix(C_21_hn_gt[pose_i], normalize=True)
        turning_angle = C_21_gt.log()[1]

        phi_errs_i = C_21_est.dot(C_21_gt.inv()).log()
        phi_errs[pose_i] = phi_errs_i

        if turning_angle > 2.*np.pi/180.:
            phi_errs_l.append(phi_errs_i/np.abs(C_21_gt.log()))

    print(phi_errs.mean(axis=0)*180./np.pi)
    print(np.array(phi_errs_l).mean(axis=0))

    for i in range(3):
        fig, ax = plt.subplots(1, 1, sharex='col', sharey='row')
        _plot_hist(phi_errs[:,i], ax)
        fig.savefig('so3_hists/' + filename + '_{}.pdf'.format(i), bbox_inches='tight')
        plt.close(fig)



def main():
    seqs = ['00', '02', '05']
    for seq in seqs:
        hydranet_output_file = '../fusion/hydranet_output_reverse_model_seq_{}.pt'.format(seq)
        make_histogram(hydranet_output_file, '{}_hist'.format(seq))


if __name__ == '__main__':
    # import cProfile, pstats
    # cProfile.run("run_svo()", "{}.profile".format(__file__))
    # s = pstats.Stats("{}.profile".format(__file__))
    # s.strip_dirs()
    # s.sort_stats("cumtime").print_stats(25)
    np.random.seed(14)
    main()
