#!/bin/bash

python3 crop_images.py --rootdir /home/data/SARShip \
                        --savedir /home/data/SARShipDataset

# 根据分辨率切分
#python3 crop_images.py --rootdir /home/data/SARShip \
#                        --savedir /home/data/SARShipDataset \
#                        --split_res

# 根据近岸远海切分
#python3 crop_images.py --rootdir /home/data/SARShip \
#                        --savedir /home/data/SARShipDataset \
#                        --split_shore
