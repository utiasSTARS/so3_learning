import numpy as np
import torch

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


def get_image_paths(data_path, trial_str, pose_deltas, img_type='rgb', eval_type='train', add_reverse=False):
    if img_type == 'rgb':
        impath_l = os.path.join(data_path, 'image_02', 'data', '*.png')
        impath_r = os.path.join(data_path, 'image_03', 'data', '*.png')
    elif img_type == 'mono':
        impath_l = os.path.join(data_path, 'image_00', 'data', '*.png')
        impath_r = os.path.join(data_path, 'image_01', 'data', '*.png')
    else:
        raise ValueError('img_type must be `rgb` or `mono`')

    imfiles_l = sorted(glob.glob(impath_l))
    imfiles_r = sorted(glob.glob(impath_r))

    imfiles_l = [imfiles_l[i] for i in KITTI_SEQS_DICT[trial_str]['frames']]
    imfiles_r = [imfiles_r[i] for i in KITTI_SEQS_DICT[trial_str]['frames']]

    image_paths = []
    for p_delta in pose_deltas:
        if eval_type == 'train':
            image_paths.extend([[imfiles_l[i], imfiles_r[i], imfiles_l[i + p_delta], imfiles_r[i + p_delta]] for i in
                                range(len(imfiles_l) - p_delta)])
            if add_reverse:
                image_paths.extend(
                    [[imfiles_l[i + p_delta], imfiles_r[i + p_delta], imfiles_l[i], imfiles_r[i]] for i in
                     range(len(imfiles_l) - p_delta)])

        elif eval_type == 'test':
            # Only add every p_delta'th quad
            image_paths.extend([[imfiles_l[i], imfiles_r[i], imfiles_l[i + p_delta], imfiles_r[i + p_delta]] for i in
                                range(0, len(imfiles_l) - p_delta, p_delta)])

        print('Adding {} image quads from trial {} for pose_delta: {}. Total quads: {}.'.format(img_type, trial_str,
                                                                                                p_delta,
                                                                                                len(image_paths)))
    return image_paths

# Load datasets
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])