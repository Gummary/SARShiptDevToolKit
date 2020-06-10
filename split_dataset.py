import sys
import glob
import random
import os
import shutil
import logging
import argparse

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.setLevel(logging.INFO)


def check_file_exist(path):
    return os.path.isfile(path)


def create_dataset(dataset_root, annotations, annotation_src, image_src):
    logger.info('=> Create dataset in {}'.format(dataset_root))
    ANNOTATION_PATH = "Annotations"
    IMAGE_PATH = "JPEGImages"
    img_dst = os.path.join(dataset_root, IMAGE_PATH)
    anno_dst = os.path.join(dataset_root, ANNOTATION_PATH)

    create_dirs([img_dst, anno_dst])

    for anno in annotations:
        img_name = anno.replace("xml", "png")
        assert check_file_exist(os.path.join(image_src, img_name)), "{} doesn't exists".format(img_name)
        src = os.path.join(annotation_src, anno)
        dst = os.path.join(anno_dst, anno)
        shutil.copyfile(src, dst)

        src = os.path.join(image_src, img_name)
        dst = os.path.join(img_dst, img_name)
        shutil.copyfile(src, dst)


def create_dirs(dirs):
    if type(dirs) is not list:
        create_dirs([dirs])
    for name in dirs:
        os.makedirs(name, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_root", help="The root path of the dataset")
    parser.add_argument('--annodir', help="The annonation path under the dataset root", default='labeltxt')
    parser.add_argument('--imgdir', help="The images path under the dataset root", default='images')
    parser.add_argument('--valid_size', help="Validation size", default=0.2, type=float)

    parser.add_argument('--shuffle', help='Shuffle the dataset before split', action='store_false')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    for k, v in args.__dict__.items():
        logger.info('{} {}'.format(k, v))


    dataset_root = args.dataset_root
    annotation_src = os.path.join(dataset_root, args.annodir)
    image_src = os.path.join(dataset_root, args.imgdir)
    valid_size = args.valid_size

    all_annos = [os.path.basename(f) for f in glob.glob("{0}/*.xml".format(annotation_src))]

    if args.shuffle:
        random.shuffle(all_annos)

    num_annos = len(all_annos)
    train_size = int(num_annos * (1 - valid_size))
    train_annos = all_annos[:train_size]
    valid_annos = all_annos[train_size:]

    create_dataset(os.path.join(dataset_root, 'Train'), train_annos, annotation_src, image_src)
    create_dataset(os.path.join(dataset_root, 'Valid'), valid_annos, annotation_src, image_src)

    with open(os.path.join(dataset_root, "train_list.txt"), 'w') as f:
        f.writelines('\n'.join([str(x.split('.')[0]) for x in train_annos]))

    with open(os.path.join(dataset_root, "valid_list.txt"), "w") as f:
        f.writelines('\n'.join([str(x.split('.')[0]) for x in valid_annos]))


if __name__ == '__main__':
    main()
