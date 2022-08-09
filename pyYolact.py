import sys
import subprocess
import time
import cv2
import math

sys.path.insert(1, '/home/sunny/yolact')


def yolact_main(FileTime):
    print("start")
    start = time.time()
    value = subprocess.check_output(["python3", "/home/sunny/yolact/eval.py",
                                     '--trained_model', '/home/sunny/yolact/weights/yolact_base_567_710000.pth',
                                     '--score_threshold', '0.90',
                                     '--top_k', '30',
                                     '--image',
                                     '/home/sunny/wb_data/{}_0_Color.jpg:/home/sunny/wb_data/{}_1_Mask.jpg'.format(
                                         FileTime, FileTime)
                                     ])
    end = time.time()
    # print("value:", value)
    # print(value.decode("utf-8"))
    box_list = eval(value.decode("utf-8").split("box_list")[1])
    print("box num:", len(box_list))

    # print("box 0:", box_list[0])
    # print("box 0:", box_list[0][0])
    # print("box 0:", box_list[0][0][0])
    img = cv2.imread("/home/sunny/wb_data/{}_1_Mask.jpg".format(FileTime))
    box_center = []
    for i in range(len(box_list)):
        degree = box_list[i][2]
        dir = box_list[i][3]
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, "0", (box_list[i][0][0][0], box_list[i][0][0][1]), font, 0.5, (0, 0, 255), 2)
        cv2.putText(img, "1", (box_list[i][0][1][0], box_list[i][0][1][1]), font, 0.5, (0, 0, 255), 2)
        cv2.putText(img, "2", (box_list[i][0][2][0], box_list[i][0][2][1]), font, 0.5, (0, 0, 255), 2)
        cv2.putText(img, "3", (box_list[i][0][3][0], box_list[i][0][3][1]), font, 0.5, (0, 0, 255), 2)
        cv2.putText(img, str(degree), (box_list[i][1][0], box_list[i][1][1]), font, 0.5, (0, 0, 255), 2)
        cv2.putText(img, str(dir), (box_list[i][1][0], box_list[i][1][1] - 20), font, 0.5, (255, 0, 0), 2)
        box_center.append([box_list[i][1][0], box_list[i][1][1], dir])
    cv2.imwrite("/home/sunny/wb_data/{}_2_Circle.jpg".format(FileTime), img)
    # box_list = str(value).split("box_list")[1].replace("\\n", "")
    # print()
    # print("box_list:")
    # print(box_list)
    # print()
    print("YOLACT time: {:.2f}".format(end - start))

    box = [s for s in box_center if 550 > s[0] > 240]

    return box


if __name__ == '__main__':
    print("target:", yolact_main("0719_160208"))
