import sys
sys.path.insert(1, '/home/yolact')

# from eval import parse_args
import subprocess
import time
import cv2
def yolact_main(FileTime):
    print("start")
    start = time.time()
    value = subprocess.check_output(["python3", "/home/yolact/eval.py",
                     '--trained_model', '/home/yolact/weights/yolact_0701_16_170000.pth',
                     '--score_threshold', '0.90',
                     '--top_k', '30',
                     '--image', '/home/wb_data/{}_0_Color.jpg:/home/wb_data/{}_1_Mask.jpg'.format(FileTime, FileTime)
                     ])
    end = time.time()
    # print("value:", value)
    # print(value.decode("utf-8"))
    box_list = eval(value.decode("utf-8").split("box_list")[1])
    print("box num:", len(box_list))

    # print("box 0:", box_list[0])
    # print("box 0:", box_list[0][0])
    # print("box 0:", box_list[0][0][0])
    img = cv2.imread("/home/wb_data/{}_1_Mask.jpg".format(FileTime))
    box_center = []
    for i in range(len(box_list)):
        cv2.circle(img, (box_list[i][0][0][0], box_list[i][0][0][1]), 5, (0, 0, 255), 3)
        cv2.circle(img, (box_list[i][0][1][0], box_list[i][0][1][1]), 5, (0, 0, 255), 3)
        cv2.circle(img, (box_list[i][0][2][0], box_list[i][0][2][1]), 5, (0, 0, 255), 3)
        cv2.circle(img, (box_list[i][0][3][0], box_list[i][0][3][1]), 5, (0, 0, 255), 3)
        cv2.circle(img, (box_list[i][1][0], box_list[i][1][1]), 5, (0, 0, 255), 3)
        box_center.append(box_list[i][1])
    cv2.imwrite("/home/wb_data/{}_2_Circle.jpg".format(FileTime), img)
    # box_list = str(value).split("box_list")[1].replace("\\n", "")
    # print()
    # print("box_list:")
    # print(box_list)
    # print()
    # print("YOLACT time: {:.2f}".format(end - start))
    return box_center


if __name__ == '__main__':
    yolact_main()
