# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 09:25:16 2019

@author: MagicMicky
https://blog.csdn.net/coooo0l/article/details/84492916
"""
import xml.etree.ElementTree as ET
import pickle
import os
from os import getcwd
import numpy as np
from PIL import Image

import imgaug as ia
from imgaug import augmenters as iaa

ia.seed(1)


def read_xml_annotation(root, image_id):
    in_file = open(os.path.join(root, image_id))
    tree = ET.parse(in_file)
    root = tree.getroot()
    bndboxlist = []

    for object in root.findall('object'):
        bndbox = object.find('bndbox')

        xmin = int(bndbox.find('xmin').text)
        xmax = int(bndbox.find('xmax').text)
        ymin = int(bndbox.find('ymin').text)
        ymax = int(bndbox.find('ymax').text)
        # print(xmin,ymin,xmax,ymax)
        bndboxlist.append([xmin, ymin, xmax, ymax])
        # print(bndboxlist)

    bndbox = root.find('object').find('bndbox')
    return bndboxlist


# (506.0000, 330.0000, 528.0000, 348.0000) -> (520.4747, 381.5080, 540.5596, 398.6603)
def change_xml_annotation(root, image_id, new_target):
    new_xmin = new_target[0]
    new_ymin = new_target[1]
    new_xmax = new_target[2]
    new_ymax = new_target[3]

    in_file = open(os.path.join(root, str(image_id) + '.xml'))
    tree = ET.parse(in_file)
    xmlroot = tree.getroot()
    object = xmlroot.find('object')
    bndbox = object.find('bndbox')
    xmin = bndbox.find('xmin')
    xmin.text = str(new_xmin)
    ymin = bndbox.find('ymin')
    ymin.text = str(new_ymin)
    xmax = bndbox.find('xmax')
    xmax.text = str(new_xmax)
    ymax = bndbox.find('ymax')
    ymax.text = str(new_ymax)
    tree.write(os.path.join(root, str(image_id) + "_aug" + '.xml'))


def change_xml_list_annotation(root, image_id, new_target, saveroot, id):
    in_file = open(os.path.join(root, str(image_id) + '.xml'))
    tree = ET.parse(in_file)
    xmlroot = tree.getroot()
    index = 0

    for object in xmlroot.findall('object'):
        bndbox = object.find('bndbox')

        new_xmin = new_target[index][0]
        new_ymin = new_target[index][1]
        new_xmax = new_target[index][2]
        new_ymax = new_target[index][3]

        xmin = bndbox.find('xmin')
        xmin.text = str(new_xmin)
        ymin = bndbox.find('ymin')
        ymin.text = str(new_ymin)
        xmax = bndbox.find('xmax')
        xmax.text = str(new_xmax)
        ymax = bndbox.find('ymax')
        ymax.text = str(new_ymax)

        index = index + 1

    # fix file name
    filename = xmlroot.find('filename')
    filename.text = str(image_id) + "_aug_" + str(id) + '.jpg'

    tree.write(os.path.join(saveroot, str(image_id) + "_aug_" + str(id) + '.xml'))


def mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        print(path + 'build done')
        return True
    else:
        print(path + ' ok')
        return False


if __name__ == "__main__":

    IMG_DIR = "/home/wb_data/2160p"
    XML_DIR = "/home/wb_data/2160p_xml"

    AUG_IMG_DIR = "/home/wb_data/2160p_aug"
    mkdir(AUG_IMG_DIR)
    AUG_XML_DIR = "/home/wb_data/2160p_aug"
    mkdir(AUG_XML_DIR)

    AUGLOOP = 100

    boxes_img_aug_list = []
    new_bndbox = []
    new_bndbox_list = []

    seq = iaa.Sequential([
        # iaa.Flipud(0.5),  # vertically flip 20% of all images
        # iaa.Fliplr(0.5),
        iaa.Multiply((0.7, 1.4)),  # , per_channel=0.1),  # change brightness, doesn't affect BBs
        iaa.GaussianBlur(sigma=(0, 0.5)),  # iaa.GaussianBlur(0.5),
        iaa.Affine(
            translate_percent={"x": 0.02, "y": 0.02},
            #            translate_px={"x": 20, "y": 20},
            scale=(0.8, 1.1),
            # rotate=(-2, 2)
        ),  # translate by 40/60px on x/y axis, and scale to 50-70%, affects BBs
        iaa.AdditiveGaussianNoise(loc=0, scale=(0, 0.03*255)),
        iaa.Sharpen(alpha=(0, 0.2), lightness=(0.8, 1.2)),  #
        #        iaa.CoarseDropout(0.01, size_percent=0.1)
        #        iaa.ContrastNormalization((0.8, 1.2)),
        iaa.Add((-25, 25)),
        iaa.Rot90((1, 3), keep_size=False)

    ], random_order=True)

    for root, sub_folders, files in os.walk(XML_DIR):

        for name in files:

            bndbox = read_xml_annotation(XML_DIR, name)

            for epoch in range(AUGLOOP):
                seq_det = seq.to_deterministic()

                img = Image.open(os.path.join(IMG_DIR, name[:-4] + '.jpg'))
                img = np.array(img)

                for i in range(len(bndbox)):
                    bbs = ia.BoundingBoxesOnImage([
                        ia.BoundingBox(x1=bndbox[i][0], y1=bndbox[i][1], x2=bndbox[i][2], y2=bndbox[i][3]),
                    ], shape=img.shape)

                    bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]
                    boxes_img_aug_list.append(bbs_aug)

                    # new_bndbox_list:[[x1,y1,x2,y2],...[],[]]
                    new_bndbox_list.append([int(bbs_aug.bounding_boxes[0].x1),
                                            int(bbs_aug.bounding_boxes[0].y1),
                                            int(bbs_aug.bounding_boxes[0].x2),
                                            int(bbs_aug.bounding_boxes[0].y2)])

                image_aug = seq_det.augment_images([img])[0]
                path = os.path.join(AUG_IMG_DIR, str(name[:-4]) + "_aug_" + str(epoch) + '.jpg')
                image_auged = bbs.draw_on_image(image_aug, size=0)
                Image.fromarray(image_auged).save(path)

                change_xml_list_annotation(XML_DIR, name[:-4], new_bndbox_list, AUG_XML_DIR, epoch)
                print(str(name[:-4]) + "_aug_" + str(epoch) + '.jpg')
                new_bndbox_list = []
