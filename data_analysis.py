import glob
import os
import os.path as osp
import cv2
import xml.etree.ElementTree as ET
import numpy as np

import matplotlib.pyplot as plt



ANNOTATION_SRC = "labeltxt"
IMAGE_SRC = "images"
all_annos = [os.path.basename(f) for f in glob.glob("{0}/*.xml".format(ANNOTATION_SRC))]
theta_list = []
aspect_ratio = []

for anno_name in all_annos:
    img_id = osp.splitext(anno_name)[0]

    xml_path = osp.join(ANNOTATION_SRC,'{}.xml'.format(img_id))
    tree = ET.parse(xml_path)
    root = tree.getroot()
    bboxes = []
    labels = []
    bboxes_ignore = []
    labels_ignore = []
    
    for obj in root.findall('object'):
        name = obj.find('name').text
        difficult = int(obj.find('difficult').text)
        bnd_box = obj.find('bndbox')
        bbox = []
        for i in range(4):
            x = int(bnd_box.find("x{}".format(i)).text)
            y = int(bnd_box.find("y{}".format(i)).text)
            bbox.append([x,y])
        
        bbox = np.array(bbox)
        rect = cv2.minAreaRect(bbox)
        theta_list.append(rect[2])
        aspect_ratio.append(max(rect[1])//min(rect[1]))
    
# plt.hist(theta_list, bins=50)
# plt.savefig("theta.png")
plt.hist(aspect_ratio, bins=10)
plt.savefig("aspect_ratio.png")