# Copyright 2020 by He Peng (sdythp@gmail.com).
# All rights reserved.
# Licensed under the MIT License.

from collections import defaultdict
import os

def parse_file_info(file_info_path):
    """

    :param file_info_path: file info path
    :return: dict:
                all: all the images
                offshore: the offshore images
                inshore: the inshore images
                3: 3m sar resolution
                1: 1m sar resolution
    """
    assert os.path.exists(file_info_path), "File info not exists"
    with open(file_info_path, 'r', encoding='utf-8') as f:
        # Skip first comment line
        ret = defaultdict(list)
        for line in f:
            if line.startswith('#'):
                continue
            fileinfo = line.strip().split(' ')
            filename = fileinfo[0]
            shore = fileinfo[3]
            resolution = fileinfo[4]
            ret['all'].append(filename)
            ret[shore].append(filename)
            ret[resolution].append(filename)
    return ret
