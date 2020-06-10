import glob
import os
import os.path as osp
import cv2
import xml.etree.ElementTree as ET
import numpy as np


ANNOTATION_SRC = "labeltxt"
IMAGE_SRC = "images"
all_annos = [os.path.basename(f) for f in glob.glob("{0}/*.xml".format(ANNOTATION_SRC))]


for anno_name in all_annos:
    img_id = osp.splitext(anno_name)[0]
    img = cv2.imread(osp.join(IMAGE_SRC, '{}.png'.format(img_id)))

    # img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

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
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(img,[box],0,(0,191,255),2)
            
        # cv2.rectangle(img, (xmin, ymin), (xmax, ymax), 255, 2)
    print(osp.join(IMAGE_SRC, "{0}_gt.png".format(img_id)))
    cv2.imwrite(osp.join(IMAGE_SRC, "{0}_gt.png".format(img_id)), img)