import numpy as np
# from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d  # noqa: F401

import pyk4a
from pyk4a import Config, PyK4A
import sys
sys.path.append('/home/pyk4a/example')
from helpers import colorize

import cv2
import time

import pyYolo
import sortlist

def main():
    root = "/home/wb_data/"
    k4a = PyK4A(
        Config(
            color_resolution=pyk4a.ColorResolution.RES_1080P,  # RES_1080P
            camera_fps=pyk4a.FPS.FPS_5,
            depth_mode=pyk4a.DepthMode.WFOV_2X2BINNED,
            synchronized_images_only=True,
        )
    )
    k4a.start()

    k4a.whitebalance = 4000
    assert k4a.whitebalance == 4000
    # k4a.whitebalance = 4010
    # assert k4a.whitebalance == 4010

    HW = [1080, 1920]
    ROIcut = [320, 440]
    ROI = [HW[0] // 2 - ROIcut[0], HW[0] // 2 + ROIcut[0], HW[1] // 2 - ROIcut[1], HW[1] // 2 + ROIcut[1]]  # [y,x]

    while True:
        totaltime = time.time()
        while 1:
            f = open('/home/rs_waterbox/CameraCMD.txt', 'r')
            CameraCMD = f.readlines()[0].split('\n')[0]
            if CameraCMD == "1":
                f.close()
                break
            time.sleep(0.2)
        FileTime = time.strftime("%m%d_%H%M%S", time.localtime())
        start1 = time.time()
        print("########## " + FileTime + " ##########")
        f = open('/home/darknet/wb/YoloList.txt', 'w')
        seq = root + "{}_Color.jpg".format(FileTime)
        f.write(seq)
        f.close()

        while True:
            time.sleep(2)
            capture = k4a.get_capture()
            if np.any(capture.depth) and np.any(capture.color):
                break
        # while True:
        #     capture = k4a.get_capture()
        #     if np.any(capture.depth) and np.any(capture.color):
        #         break

        # cv2.imwrite("/home/2160p2.jpg".format(), capture.color)
        # cv2.imwrite(root + '{}_0_capture.jpg'.format(FileTime), capture.color)
        save_img = capture.color[ROI[0]:ROI[1], ROI[2]:ROI[3]]
        cv2.imwrite(root + '{}_0_Color.jpg'.format(FileTime), save_img)
        # break
        end1 = time.time()
        CamTime = round(end1 - start1, 2)

        start2 = time.time()
        img = cv2.imread(root + '{}_0_Color.jpg'.format(FileTime))
        wbmid, water = pyYolo.main(img, FileTime)
        end2 = time.time()
        YoloTime = round(end2 - start2, 2)

        start3 = time.time()
        num_wbmid, num_water, box = sortlist.wb_sort(wbmid, water)
        print("box:", box)

        # print("capture:", capture.transformed_depth_point_cloud[1104][1917])  # 500:1700   1100:2700
        # print(capture.transformed_depth_point_cloud.shape)
        # print(box[0][0], box[0][1], ROI[0], ROI[1], ROI[2], ROI[3])  # 774 455
        # print("capture:", capture.transformed_depth_point_cloud[box[0][1]+ROI[0]][box[0][0]+ROI[2]])  # 500:1700   1100:2700



        if num_wbmid != num_water:
            cv2.imwrite(root + "img_fix/" + '{}_0_Color.jpg'.format(FileTime), save_img)
        if box == 0:
            f = open(root + '{}_4_Robot.txt'.format(FileTime), 'w')
            f.write("None" + " ")
            f.write("None" + " ")
            f.write("None" + " ")
            f.write("None" + " ")
            f.write(str(num_wbmid))
            f.close()
        else:
            box_xyz = []
            for i in box:
                # box_z = int(transformed_depth_image[i[1] + ROI[0]][i[0] + ROI[2]])
                box_z = capture.transformed_depth_point_cloud[i[1]+ROI[0]][i[0]+ROI[2]][2]
                box_xyz.append(i[:] + [box_z])

            box_xyz = sorted(box_xyz, key=lambda m: (m[5]), reverse=0)
            print("All Z:  ", box_xyz)
            box_xyz = [s for s in box_xyz if (s[5] - box_xyz[0][5]) ** 2 < 40000]
            print("Same Z: ", box_xyz)
            box_xyz = sorted(box_xyz, key=lambda m: (m[1]), reverse=1)
            print("Sort Y: ", box_xyz)
            box_xyz = [s for s in box_xyz if (box_xyz[0][1] - s[1]) ** 2 < 1000]
            print("Same Y: ", box_xyz)
            box_xyz = sorted(box_xyz, key=lambda m: (m[0]), reverse=1)
            print("Sort X: ", box_xyz)

            target = box_xyz[0]
            print("Pixel target", target)
            # print("            ", [box[0][1] + ROI[0]][box[0][0] + ROI[2]])

            cv2.circle(save_img, (target[0], target[1]), 6, (0, 0, 255), 2)
            cv2.rectangle(save_img, (70, 0), (835, 639), (0, 0, 255), 1)
            cv2.imwrite(root + '{}_3_Tar.jpg'.format(FileTime), save_img)
            cv2.imwrite(root + '{}_2_Depth.jpg'.format(FileTime), colorize(capture.transformed_depth, (None, 50000)))

            robot_target = capture.transformed_depth_point_cloud[target[1]+ROI[0]][target[0]+ROI[2]]
            target = [robot_target[0], -robot_target[1], target[4], robot_target[2]]
            print("Robot target", target)
            # print("capture[704][986] :", capture.transformed_depth_point_cloud[704][986])
            # print("capture[551][1176]:", capture.transformed_depth_point_cloud[551][1176])
            # for i in range(252, 804):
            #     for j in range(626, 1318):
            #         xy = capture.transformed_depth_point_cloud[i][j]
            #         if xy[0] == 0 and xy[1] == 0:
            #             print(i, j)
            #             break




            f = open(root + '{}_4_Robot.txt'.format(FileTime), 'w')
            f.write(str(target[0]) + " ")
            f.write(str(target[1]) + " ")
            f.write(str(target[2]) + " ")
            f.write(str(target[3]) + " ")
            f.write(str(num_wbmid))
            f.close()


            # points = capture.depth_point_cloud.reshape((-1, 3))
            # colors = capture.transformed_color[..., (2, 1, 0)].reshape((-1, 3))
            # k4a.stop()
        end3 = time.time()
        SortTime = round(end3 - start3, 2)

        f = open(root + '{}_5_Time.txt'.format(FileTime), 'w')
        f.write("CamTime: " + str(CamTime) + "\n")
        f.write("YoloTime: " + str(YoloTime) + "\n")
        f.write("SortTime: " + str(SortTime) + "\n")
        f.close()

        totaltimeend = time.time()
        print("CamTime:  ", CamTime)
        print("YoloTime: ", YoloTime)
        print("SortTime: ", SortTime)
        print("TotalTime:", round(totaltimeend - totaltime, 2))
        # k4a.stop()
        # break # [-318, -309, 0, 1876] [660, 483, 692, 482, 0, 1876]
        f = open('/home/rs_waterbox/CameraCMD.txt', 'w')
        f.write("2")
        f.close()
        time.sleep(2)



if __name__ == "__main__":
    # FileTime = time.strftime("%m%d_%H%M%S", time.localtime())
    main()
