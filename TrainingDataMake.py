# -*- coding: utf-8 -*-


import os
import random

IMG_DIR = "/home/wb_data/2160p_aug"
# XML_DIR = "/home/sunny/darknet/wb35_xml"
jpgfile = []
for file in os.listdir(IMG_DIR):
    if file.endswith(".jpg"):
        # print(os.path.join(IMG_DIR, file))
        jpgfile.append(os.path.join(IMG_DIR, file))
# testList = random.sample(os.listdir(IMG_DIR), int(len(os.listdir(IMG_DIR))*0.2))
testList = random.sample(jpgfile, int(len(jpgfile)*0.2))
total = os.listdir(IMG_DIR)
print('len in folder', len(os.listdir(IMG_DIR)))
print('all jpg', jpgfile)
print('len jpg', len(jpgfile))
print('len testList', testList)
print('len testList', len(testList))

trainList = [x for x in jpgfile if x not in testList]
print('trainList', trainList)
print('total trainList', len(trainList))

ftest = open("/home/darknet/wb_2160p_test.txt", "a")
for fname in testList:
    srcpath = os.path.join(IMG_DIR, fname)
    print(srcpath)
    ftest.write(srcpath+"\n")
ftest.close()

ftrain = open("/home/darknet/wb_2160p_train.txt", "a")
for fname in trainList:
    srcpath = os.path.join(IMG_DIR, fname)
    print(srcpath)
    ftrain.write(srcpath+"\n")
ftrain.close()

