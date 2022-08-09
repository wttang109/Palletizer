import glob
import os
import random
import shutil

root = "/home/sunny/c930e"
os.chdir(root)

ImageNames = []
for file in glob.glob("*.jpg"):
    ImageNames.append(file.split(".")[0])
print(ImageNames)
print(len(ImageNames))

ratio = int(len(ImageNames) * 0.2)

TestFile = random.sample(ImageNames, ratio)
print(TestFile)

os.makedirs(root + '/train')
os.makedirs(root + '/test')

for i in TestFile:
    shutil.move(root + '/{}.jpg'.format(i), root + '/test')
    shutil.move(root + '/{}.json'.format(i), root + '/test')

TrainFile = []
for file in glob.glob("*.jpg"):
    shutil.move(root + '/{}.jpg'.format(file.split(".")[0]), root + '/train')
    shutil.move(root + '/{}.json'.format(file.split(".")[0]), root + '/train')
