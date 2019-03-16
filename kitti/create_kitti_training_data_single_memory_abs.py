from pyslam.metrics import TrajectoryMetrics
import pickle, csv, glob, os
import random
import numpy as np
from liegroups.numpy import SE3
import pykitti

KITTI_SEQS_DICT = {'00': {'date': '2011_10_03',
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

def extract_global_orientations(kitti_path, seq):
    """Compute delta pose errors on VO estimates """
    C_imu_w = []
    pose_ids = []
    seqs = []

    date = seqs[seq]['date']
    drive = seqs[seq]['drive']
    frames = seqs[seq]['frames']
    dataset = pykitti.raw(kitti_path, date, drive, frames=frames)
    dataset_len = len(dataset)

    #Load ground truth IMU poses
    C_imu_w = [SE3.from_matrix(o.T_w_imu).rot.inv() for o in dataset.oxts]
    pose_ids = list(range(len(C_imu_w)))
    seqs = [seq]*len(C_imu_w)
    return (C_imu_w, pose_ids, seqs)

def process_ground_truth(trial_strs, kitti_path, eval_type='train'):

    C_imu_w_all = []
    pose_ids_all = []
    seqs_all = []

    for t_id, trial_str in enumerate(trial_strs):

        (C_imu_w, pose_ids, seqs) = extract_global_orientations(kitti_path, trial_str)

        pose_ids_all.extend(pose_ids)
        seqs_all.extend(seqs)
        C_imu_w_all.extend(C_imu_w)

    return (C_imu_w_all, pose_ids_all, seqs_all)



def main():

    #Removed 01 and 04 (road trials)
    all_trials = ['00','02','05','06', '07', '08', '09', '10','01','04']
    #all_trials = ['00', '02', '05', '06']


    #Where is the KITTI data?


    #Obelisk
    kitti_path = '/media/datasets/KITTI/raw'

    #Monolith:
    #kitti_path = '/media/m2-drive/datasets/KITTI/raw'
    #tm_path = '/media/raid5-array/experiments/Deep-PC/stereo_vo_results/baseline'

    #Where should we output the training files?
    data_path = './datasets/obelisk'

    
    for t_i, test_trial in enumerate(all_trials):
        if t_i > 2:
            break #Only produce trials for 00, 02 and 05

        if test_trial == all_trials[-1]:
            train_trials = all_trials[:-1]
        else:
            train_trials = all_trials[:t_i] + all_trials[t_i+1:]


        print('Processing.. Test: {}. Train: {}.'.format(test_trial, train_trials))

        (train_C_imu_w, train_pose_ids, train_sequences) = process_ground_truth(kitti_path, train_trials)
        print('Processed {} training poses.'.format(len(train_C_imu_w)))


        (test_C_imu_w, test_pose_ids, test_sequences) = process_ground_truth(kitti_path, [test_trial])
        print('Processed {} test poses.'.format(len(test_C_imu_w)))

        #Save the data!
        kitti_data = {}

        kitti_data['train_seqs'] = train_sequences
        kitti_data['train_pose_indices'] = train_pose_ids
        kitti_data['train_C_imu_w'] = train_C_imu_w

        kitti_data['test_seqs'] = test_sequences
        kitti_data['test_pose_indices'] = test_pose_ids
        kitti_data['test_C_imu_w'] = test_C_imu_w


        data_filename = os.path.join(data_path, 'kitti_singlefile_data_sequence_{}_abs.pickle'.format(test_trial))
        print('Saving to {} ....'.format(data_filename))

        with open(data_filename, 'wb') as f:
            pickle.dump(kitti_data, f, pickle.HIGHEST_PROTOCOL)

        print('Saved.')


if __name__ == '__main__':
    main()