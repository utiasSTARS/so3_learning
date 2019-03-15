import pykitti
import numpy as np
from liegroups import SE3, SO3
import os



def run_vio(basedir, date, drive, pose_range, metrics_filename=None):

    #Load data
    dataset = pykitti.raw(basedir, date, drive, frames=pose_range)
    dataset_len = len(dataset)


    #Load initial pose and ground truth
    T_cam_imu = SE3.from_matrix(dataset.calib.T_cam2_imu)
    T_cam_imu.normalize()
    T_w_0 = SE3.from_matrix(dataset.oxts[0].T_w_imu).dot(T_cam_imu.inv())
    T_w_0.normalize()
    T_w_c_gt = [SE3.from_matrix(o.T_w_imu).dot(T_cam_imu.inv())
        for o in dataset.oxts]

    vio = VisualInertialPipeline(dataset, T_cam_imu, T_w_c_gt)

    print(dataset.oxts[0])
    # #Compute statistics
    # T_w_c_est = [T for T in svo.T_w_c]
    # tm = TrajectoryMetrics(T_w_c_gt, T_w_c_est, convention='Twv')
    # # # Save to file
    # if metrics_filename:
    #     print('Saving to {}'.format(metrics_filename))
    #     tm.savemat(metrics_filename, extras={'Sigma_21': svo.Sigma})

    #return tm



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


    kitti_basedir = '/Users/valentinp/research/data/kitti/'


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


    compute_seqs = ['06']#, '06', '07', '08', '09', '10']
    for seq in compute_seqs:

        date = seqs[seq]['date']
        drive = seqs[seq]['drive']
        frames = seqs[seq]['frames']

        print('Odometry sequence {} | {} {}'.format(seq, date, drive))
        metrics_filename = os.path.join(date + '_drive_' + drive + '.mat')

        run_vio(kitti_basedir, date, drive, frames, metrics_filename=None)


        
# #Output CSV stats
    # csv_filename = os.path.join(export_dir, 'stats.csv')
    # with open(csv_filename, "w") as f:
    #     writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
    #     writer.writerow(["Seq", "Trans. ARMSE (m)", "Rot. ARMSE (rad)"])
    #     writer.writerows(armse_list)
    # 




if __name__ == '__main__':
    # import cProfile, pstats
    # cProfile.run("run_svo()", "{}.profile".format(__file__))
    # s = pstats.Stats("{}.profile".format(__file__))
    # s.strip_dirs()
    # s.sort_stats("cumtime").print_stats(25)
    np.random.seed(14)
    main()

