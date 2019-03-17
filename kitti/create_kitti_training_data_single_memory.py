from pyslam.metrics import TrajectoryMetrics
import pickle, csv, glob, os
import random
import numpy as np
from liegroups.numpy import SE3

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

def compute_vo_pose_errors(tm, pose_deltas, seq, eval_type='train', add_reverse=False):
    """Compute delta pose errors on VO estimates """
    T_21_gts = []
    T_21_ests = []
    pair_pose_ids = []
    seqs = []

    for p_delta in pose_deltas:

        if eval_type=='train':
            pose_ids = range(len(tm.Twv_gt) - p_delta)
        elif eval_type=='test':
            pose_ids = range(0, len(tm.Twv_gt) - p_delta, p_delta)

        for p_idx in pose_ids:
            T_21_gt = tm.Twv_gt[p_idx + p_delta].inv().dot(tm.Twv_gt[p_idx])
            T_21_est = tm.Twv_est[p_idx + p_delta].inv().dot(tm.Twv_est[p_idx])
            T_21_gts.append(T_21_gt)
            T_21_ests.append(T_21_est)
            pair_pose_ids.append([p_idx, p_idx + p_delta])
            seqs.append(seq)

        if add_reverse and eval_type=='train':
            for p_idx in pose_ids:
                T_21_gt = tm.Twv_gt[p_idx].inv().dot(tm.Twv_gt[p_idx + p_delta])
                T_21_est = tm.Twv_est[p_idx].inv().dot(tm.Twv_est[p_idx + p_delta])
                T_21_gts.append(T_21_gt)
                T_21_ests.append(T_21_est)
                pair_pose_ids.append([p_idx + p_delta, p_idx])
                seqs.append(seq)

    return (T_21_gts, T_21_ests, pair_pose_ids, seqs)

def process_ground_truth(trial_strs, tm_path, pose_deltas, eval_type='train', add_reverse=False):
    
    T_21_gt_all = []
    T_21_est_all = []
    pose_ids = []
    sequences = []
    tm_mat_files = []

    for t_id, trial_str in enumerate(trial_strs):

        tm_mat_file = os.path.join(tm_path, KITTI_SEQS_DICT[trial_str]['date'] + '_drive_' + KITTI_SEQS_DICT[trial_str]['drive'] + '.mat')

        try:
            tm = TrajectoryMetrics.loadmat(tm_mat_file)
        except FileNotFoundError:
            tm_mat_file = os.path.join(tm_path, trial_str + '.mat')
            tm = TrajectoryMetrics.loadmat(tm_mat_file)


        (T_21_gt, T_21_est, pair_pose_ids, seqs) = compute_vo_pose_errors(tm, pose_deltas, trial_str, eval_type, add_reverse)

        T_21_gt_all.extend(T_21_gt)
        T_21_est_all.extend(T_21_est)
        pose_ids.extend(pair_pose_ids)
        sequences.extend(seqs)
        tm_mat_files.extend(tm_mat_file)


    return (pose_ids, sequences, T_21_gt_all, T_21_est_all, tm_mat_files)



def main():
    # test_trials = ['00']    
    # val_trials = ['01']
    # train_trials = ['04', '02', '05', '06', '07', '08', '09', '10']
    
    #all_trials = ['00','02','05','06', '07', '08', '09', '10', '01', '04']
    #all_trials = ['00', '02', '05', '06']
    #all_trials = ['00', '01', '02', '04', '05', '06', '07', '08', '09', '10']

    train_pose_deltas = [1] #How far apart should each quad image be? (KITTI is at 10hz, can input multiple)
    test_pose_delta = 1
    add_reverse = True #Add reverse transformations

    #Where is the KITTI data?


    #Obelisk
    kitti_path = '/media/datasets/KITTI/raw'
    tm_path = '/media/datasets/KITTI/trajectory_metrics'

    #Monolith:
    #kitti_path = '/media/m2-drive/datasets/KITTI/raw'
    #tm_path = '/media/raid5-array/experiments/Deep-PC/stereo_vo_results/baseline'

    #Where should we output the training files?
    data_path = './datasets/obelisk'

    custom_training = [[['09','10'],['00', '01', '02', '04', '05', '06', '07', '08']]]
    # #for t_i, test_trial in enumerate(all_trials):
    # #    if t_i > 2:
    #         break #Only produce trials for 00, 02 and 05
    #
    #     if test_trial == all_trials[-1]:
    #         #val_trial = all_trials[-2]
    #         train_trials = all_trials[:-1]
    #     else:
    #         #val_trial = all_trials[t_i+1]
    #         train_trials = all_trials[:t_i] + all_trials[t_i+1:]

    for test_trials, train_trials in custom_training:

        print('Processing.. Test: {}. Train: {}.'.format(test_trials, train_trials))

        #(pose_ids, sequences, T_21_gt_all, T_21_est_all, tm_mat_files)

        (train_pose_ids, train_sequences, train_T_21_gt, train_T_21_est, train_tm_mat_files) = process_ground_truth(train_trials, tm_path, train_pose_deltas, 'train', add_reverse)
        print('Processed {} training poses.'.format(len(train_T_21_gt)))

        # (val_img_paths_rgb, val_corr, val_gt, val_est, val_tm_mat_file) = process_ground_truth([val_trial], tm_path, kitti_path, [test_pose_delta], 'test', add_reverse)
        # print('Processed {} validation image quads.'.format(len(val_corr)))

        (test_pose_ids, test_sequences, test_T_21_gt, test_T_21_est, test_tm_mat_files) = process_ground_truth(test_trials, tm_path, [test_pose_delta], 'test', add_reverse)
        print('Processed {} test poses.'.format(len(test_T_21_gt)))

        #Save the data!
        kitti_data = {}


        kitti_data['train_seqs'] = train_sequences
        kitti_data['train_pose_indices'] = train_pose_ids
        kitti_data['train_T_21_gt'] = train_T_21_gt
        kitti_data['train_T_21_vo'] = train_T_21_est
        kitti_data['train_tm_mat_paths'] = train_tm_mat_files
        kitti_data['train_pose_deltas'] = train_pose_deltas

        kitti_data['test_seqs'] = test_sequences
        kitti_data['test_pose_indices'] = test_pose_ids
        kitti_data['test_T_21_gt'] = test_T_21_gt
        kitti_data['test_T_21_vo'] = test_T_21_est
        kitti_data['test_tm_mat_paths'] = test_tm_mat_files
        kitti_data['test_pose_delta'] = test_pose_delta

        data_filename = os.path.join(data_path, 'kitti_singlefile_data_sequence_{}_delta_{}_reverse_{}.pickle'.format(test_trials, test_pose_delta, add_reverse))
        print('Saving to {} ....'.format(data_filename))

        with open(data_filename, 'wb') as f:
            pickle.dump(kitti_data, f, pickle.HIGHEST_PROTOCOL)

        print('Saved.')


if __name__ == '__main__':
    main()