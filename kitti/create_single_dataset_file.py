import numpy as np
import torch
import pickle, csv, glob, os
import torchvision.transforms as transforms
from PIL import Image

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


def get_image_paths(data_path, trial_str, img_type='rgb'):
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


    return (imfiles_l, imfiles_r)

def read_and_transform(img_path, transform):
    img = Image.open(img_path).convert('RGB')
    print(transform(img))
    return None #transform(img)

def save_images(image_paths_rgb, transform, img_dims, file_name):

    num_images = len(image_paths_rgb[0])
    left_image_data = torch.empty(num_images, 3, img_dims[1], img_dims[0]).float()
    right_image_data = torch.empty(num_images, 3, img_dims[1], img_dims[0]).float()

    for idx, (im_l, im_r) in enumerate(zip(*image_paths_rgb)):
        if idx%100==0:
            print(idx)
        left_image_data[idx] = read_and_transform(im_l, transform)
        right_image_data[idx] = read_and_transform(im_r, transform)

    torch.save({
        'im_l': left_image_data,
        'im_r': right_image_data
    }, file_name)

def main():
    # Obelisk
    kitti_path = '/media/datasets/KITTI/raw'
    trial_strs = ['00','02','05','06', '07', '08', '09', '10']

    # Load datasets
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                      std=[0.229, 0.224, 0.225])
    ])

    for t_id, trial_str in enumerate(trial_strs):

        drive_folder = KITTI_SEQS_DICT[trial_str]['date'] + '_drive_' + KITTI_SEQS_DICT[trial_str]['drive'] + '_sync'
        data_path = os.path.join(kitti_path, KITTI_SEQS_DICT[trial_str]['date'], drive_folder)

        image_paths_rgb = get_image_paths(data_path, trial_str, 'rgb')
        file_name = 'seq_{}.pt'.format(trial_str)
        save_images(image_paths_rgb, transform, [224, 224], file_name)


if __name__ == '__main__':
    main()