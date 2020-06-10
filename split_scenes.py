# Copyright 2020 by He Peng (sdythp@gmail.com).
# All rights reserved.
# Licensed under the MIT License.

import os
import shutil

from utils import parse_file_info

def create_subset(file_list, src, dst):
    files = os.listdir(src)
    for file_name in files:
        file_id = file_name.split('_')[0]
        if file_id in file_list:
            shutil.copy(os.path.join(src, file_name),
                        os.path.join(dst, file_name))




def main():
    file_info_path = 'fileinfo.txt'
    base_save_idr = '/home/data/SARShipDataset'

    images = parse_file_info(file_info_path)
    all_path = os.path.join(base_save_idr, 'all')
    src_dirs = [
        os.path.join(all_path, 'Train', 'Annotations'),
        os.path.join(all_path, 'Train', 'JPEGImages'),
        os.path.join(all_path, 'Valid', 'Annotations'),
        os.path.join(all_path, 'Valid', 'JPEGImages'),

    ]

    for key, value in images.items():
        if key == 'all':
            continue
        scene = key
        scene_path = os.path.join(base_save_idr, scene)
        dst_dirs = [
            os.path.join(scene_path, 'Train', 'Annotations'),
            os.path.join(scene_path, 'Train', 'JPEGImages'),
            os.path.join(scene_path, 'Valid', 'Annotations'),
            os.path.join(scene_path, 'Valid', 'JPEGImages'),
        ]
        for dir in dst_dirs:
            os.makedirs(dir)

        file_list = [
            'SARShip{}'.format(name.split('-')[-1])
            for name in value
        ]

        for src, dst in zip(src_dirs, dst_dirs):
            print("Copy files from {} to {}".format(src, dst))
            create_subset(file_list, src, dst)



if __name__ == '__main__':
    main()