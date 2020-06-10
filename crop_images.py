import sys
import os
import matplotlib.pyplot as plt
from xml.dom.minidom import Document
import numpy as np
import copy
import cv2
import argparse
import logging

from utils import parse_file_info

logger = logging.getLogger(__name__)


def save_to_xml(save_path, im_height, im_width, objects_axis, label_name):
    im_depth = 0
    object_num = len(objects_axis)
    doc = Document()

    annotation = doc.createElement('annotation')
    doc.appendChild(annotation)

    folder = doc.createElement('folder')
    folder_name = doc.createTextNode('VOC2007')
    folder.appendChild(folder_name)
    annotation.appendChild(folder)

    filename = doc.createElement('filename')
    filename_name = doc.createTextNode('000024.jpg')
    filename.appendChild(filename_name)
    annotation.appendChild(filename)

    source = doc.createElement('source')
    annotation.appendChild(source)

    database = doc.createElement('database')
    database.appendChild(doc.createTextNode('The VOC2007 Database'))
    source.appendChild(database)

    annotation_s = doc.createElement('annotation')
    annotation_s.appendChild(doc.createTextNode('PASCAL VOC2007'))
    source.appendChild(annotation_s)

    image = doc.createElement('image')
    image.appendChild(doc.createTextNode('flickr'))
    source.appendChild(image)

    flickrid = doc.createElement('flickrid')
    flickrid.appendChild(doc.createTextNode('322409915'))
    source.appendChild(flickrid)

    owner = doc.createElement('owner')
    annotation.appendChild(owner)

    flickrid_o = doc.createElement('flickrid')
    flickrid_o.appendChild(doc.createTextNode('knautia'))
    owner.appendChild(flickrid_o)

    name_o = doc.createElement('name')
    name_o.appendChild(doc.createTextNode('gum'))
    owner.appendChild(name_o)

    size = doc.createElement('size')
    annotation.appendChild(size)
    width = doc.createElement('width')
    width.appendChild(doc.createTextNode(str(im_width)))
    height = doc.createElement('height')
    height.appendChild(doc.createTextNode(str(im_height)))
    depth = doc.createElement('depth')
    depth.appendChild(doc.createTextNode(str(im_depth)))
    size.appendChild(width)
    size.appendChild(height)
    size.appendChild(depth)
    segmented = doc.createElement('segmented')
    segmented.appendChild(doc.createTextNode('0'))
    annotation.appendChild(segmented)
    for i in range(object_num):
        objects = doc.createElement('object')
        annotation.appendChild(objects)
        object_name = doc.createElement('name')
        object_name.appendChild(doc.createTextNode(
            label_name[int(objects_axis[i][-1])]))
        objects.appendChild(object_name)
        pose = doc.createElement('pose')
        pose.appendChild(doc.createTextNode('Unspecified'))
        objects.appendChild(pose)
        truncated = doc.createElement('truncated')
        truncated.appendChild(doc.createTextNode('1'))
        objects.appendChild(truncated)
        difficult = doc.createElement('difficult')
        difficult.appendChild(doc.createTextNode('0'))
        objects.appendChild(difficult)
        bndbox = doc.createElement('bndbox')
        objects.appendChild(bndbox)

        x0 = doc.createElement('x0')
        x0.appendChild(doc.createTextNode(str((objects_axis[i][0]))))
        bndbox.appendChild(x0)
        y0 = doc.createElement('y0')
        y0.appendChild(doc.createTextNode(str((objects_axis[i][1]))))
        bndbox.appendChild(y0)

        x1 = doc.createElement('x1')
        x1.appendChild(doc.createTextNode(str((objects_axis[i][2]))))
        bndbox.appendChild(x1)
        y1 = doc.createElement('y1')
        y1.appendChild(doc.createTextNode(str((objects_axis[i][3]))))
        bndbox.appendChild(y1)

        x2 = doc.createElement('x2')
        x2.appendChild(doc.createTextNode(str((objects_axis[i][4]))))
        bndbox.appendChild(x2)
        y2 = doc.createElement('y2')
        y2.appendChild(doc.createTextNode(str((objects_axis[i][5]))))
        bndbox.appendChild(y2)

        x3 = doc.createElement('x3')
        x3.appendChild(doc.createTextNode(str((objects_axis[i][6]))))
        bndbox.appendChild(x3)
        y3 = doc.createElement('y3')
        y3.appendChild(doc.createTextNode(str((objects_axis[i][7]))))
        bndbox.appendChild(y3)

    f = open(save_path, 'w')
    f.write(doc.toprettyxml(indent=''))
    f.close()


class_list = ['ship']


def format_label(txt_list):
    format_data = []
    for i in txt_list:
        box_label = i.strip().split(",")
        if len(i) < 10:
            continue
        format_data.append(
            [int(xy) for xy in box_label[:8]] +
            [class_list.index(box_label[8])]
            # {'x0': int(i.split(' ')[0]),
            # 'x1': int(i.split(' ')[2]),
            # 'x2': int(i.split(' ')[4]),
            # 'x3': int(i.split(' ')[6]),
            # 'y1': int(i.split(' ')[1]),
            # 'y2': int(i.split(' ')[3]),
            # 'y3': int(i.split(' ')[5]),
            # 'y4': int(i.split(' ')[7]),
            # 'class': class_list.index(i.split(' ')[8]) if i.split(' ')[8] in class_list else 0,
            # 'difficulty': int(i.split(' ')[9])}
        )
        if box_label[8] not in class_list:
            logger.error('warning found a new label :', i.split(' ')[8])
            exit()
    return np.array(format_data)


def clip_image(file_idx, image, height, width, stride, boxes_all, save_dir):
    if len(boxes_all) > 0:
        shape = image.shape
        for start_h in range(0, shape[0], stride):
            for start_w in range(0, shape[1], stride):
                boxes = copy.deepcopy(boxes_all)
                box = np.zeros_like(boxes_all)
                start_h_new = start_h
                start_w_new = start_w
                if start_h + height > shape[0]:
                    start_h_new = shape[0] - height
                if start_w + width > shape[1]:
                    start_w_new = shape[1] - width
                top_left_row = max(start_h_new, 0)
                top_left_col = max(start_w_new, 0)
                bottom_right_row = min(start_h + height, shape[0])
                bottom_right_col = min(start_w + width, shape[1])

                subImage = image[top_left_row:bottom_right_row,
                           top_left_col: bottom_right_col]

                box[:, 0] = boxes[:, 0] - top_left_col
                box[:, 2] = boxes[:, 2] - top_left_col
                box[:, 4] = boxes[:, 4] - top_left_col
                box[:, 6] = boxes[:, 6] - top_left_col

                box[:, 1] = boxes[:, 1] - top_left_row
                box[:, 3] = boxes[:, 3] - top_left_row
                box[:, 5] = boxes[:, 5] - top_left_row
                box[:, 7] = boxes[:, 7] - top_left_row
                box[:, 8] = boxes[:, 8]
                center_y = 0.25 * (box[:, 1] + box[:, 3] + box[:, 5] + box[:, 7])
                center_x = 0.25 * (box[:, 0] + box[:, 2] + box[:, 4] + box[:, 6])

                cond1 = np.intersect1d(np.where(center_y[:] >= 0)[
                                           0], np.where(center_x[:] >= 0)[0])
                cond2 = np.intersect1d(np.where(center_y[:] <= (bottom_right_row - top_left_row))[0],
                                       np.where(center_x[:] <= (bottom_right_col - top_left_col))[0])
                idx = np.intersect1d(cond1, cond2)

                if len(idx) > 0:
                    xml = os.path.join(save_dir, 'labeltxt', "SARShip%s_%04d_%04d.xml" % (
                        file_idx, top_left_row, top_left_col))
                    save_to_xml(
                        xml, subImage.shape[0], subImage.shape[1], box[idx, :], class_list)
                    if subImage.shape[0] > 5 and subImage.shape[1] > 5:
                        img = os.path.join(save_dir, 'images', "SARShip%s_%04d_%04d.png" % (
                            file_idx, top_left_row, top_left_col))
                        am = np.amax(subImage)
                        subImage = subImage * 255 / am
                        cv2.imwrite(img, subImage)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--rootdir", help="The root path of SARShip dataset")
    parser.add_argument("--savedir", help="The path to save the cropped dataset")
    parser.add_argument("--fileinfo", help='The path to the fileinfo', default='fileinfo.txt')
    parser.add_argument('--width', help="The width of the cropped image", default=800, type=int)
    parser.add_argument('--height', help="The height of the cropped image", default=800, type=int)
    parser.add_argument('--stride', help="The stride or overlap between two crop images", default=256, type=int)

    parser.add_argument('--split_res', help="Split the dataset into 1m resolution and 3m resolution",
                        action='store_true')
    parser.add_argument('--split_shore', help="Split the dataset into offshore and inshore", action='store_true')

    args = parser.parse_args()

    return args


def crop_image_list(image_names, save_dir, args):
    root_dir = args.rootdir
    for idx, img_name in enumerate(image_names):
        img_path = os.path.join(root_dir, img_name, "{0}.png".format(img_name))
        assert os.path.isfile(img_path), "{} not exists!".format(img_path)
        logger.info('=> Processing {}'.format(img_name))

        txt_path = os.path.join(root_dir, img_name, "SARShip-1.mbrect")
        txt_data = open(txt_path, 'r').readlines()
        box = format_label(txt_data)

        img_data = plt.imread(img_path)
        clip_image(idx + 1, img_data, boxes_all=box, height=args.height, width=args.width, stride=args.stride,
                   save_dir=save_dir)


def main():
    args = parse_args()
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    for k, v in args.__dict__.items():
        logger.info('{}: {}'.format(k, v))

    images = parse_file_info(args.fileinfo)

    base_save_dir = args.savedir
    os.makedirs(os.path.join(base_save_dir, 'all', 'labeltxt'), exist_ok=True)
    os.makedirs(os.path.join(base_save_dir, 'all', 'images'), exist_ok=True)

    if args.split_res:
        logger.info('Split dataset into 3m and 1m resolution')
        r3_path = os.path.join(base_save_dir, 'r3m')
        r1_path = os.path.join(base_save_dir, 'r1m')
        os.makedirs(os.path.join(r3_path, 'labeltxt'), exist_ok=True)
        os.makedirs(os.path.join(r3_path, 'images'), exist_ok=True)
        os.makedirs(os.path.join(r1_path, 'labeltxt'), exist_ok=True)
        os.makedirs(os.path.join(r1_path, 'images'), exist_ok=True)

        crop_image_list(images['3'], r3_path, args)
        crop_image_list(images['1'], r1_path, args)

    elif args.split_shore:
        logger.info('Split dataset into inshore and offshore')
        inshore_path = os.path.join(base_save_dir, 'inshore')
        offshore_path = os.path.join(base_save_dir, 'offshore')
        os.makedirs(os.path.join(inshore_path, 'labeltxt'), exist_ok=True)
        os.makedirs(os.path.join(inshore_path, 'images'), exist_ok=True)
        os.makedirs(os.path.join(offshore_path, 'labeltxt'), exist_ok=True)
        os.makedirs(os.path.join(offshore_path, 'images'), exist_ok=True)

        crop_image_list(images['inshore'], inshore_path, args)
        crop_image_list(images['offshore'], offshore_path, args)
    else:
        crop_image_list(images['all'], os.path.join(base_save_dir, 'all'), args)

if __name__ == '__main__':
    main()
