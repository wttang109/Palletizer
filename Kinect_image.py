import yolocmd
import pyYolo
import sortlist
import time
import math

import sys
sys.path.append('/home/pyKinectAzure/pyKinectAzure')
import numpy as np
from pyKinectAzure import pyKinectAzure, _k4a, postProcessing
import cv2

def main():  # FOV 130 115 185
    root = "/home/wb_data/"
    modulePath = '/usr/lib/x86_64-linux-gnu/libk4a.so'
    # f = open('/home/rs_waterbox/CameraCMD.txt', 'w')
    # f.write("2")
    # f.close()
    pyK4A = pyKinectAzure(modulePath)

    pyK4A.device_open()
    device_config = pyK4A.config
    device_config.color_format = _k4a.K4A_IMAGE_FORMAT_COLOR_BGRA32
    device_config.color_resolution = _k4a.K4A_COLOR_RESOLUTION_1080P  # 2160
    device_config.depth_mode = _k4a.K4A_DEPTH_MODE_WFOV_2X2BINNED
    device_config.calibration_type = _k4a.K4A_CALIBRATION_TYPE_COLOR
    # print(device_config)

    # Start cameras using modified configuration
    pyK4A.device_start_cameras(device_config)
    HW = [1080, 1920]
    ROIcut = [320, 440]
    ROI = [HW[0] // 2 - ROIcut[0], HW[0] // 2 + ROIcut[0], HW[1] // 2 - ROIcut[1], HW[1] // 2 + ROIcut[1]]  # [y,x]
    while 1:
        pyK4A.device_get_capture()
        depth_image_handle = pyK4A.capture_get_depth_image()
        color_image_handle = pyK4A.capture_get_color_image()
        if depth_image_handle and color_image_handle:
            break
    pyK4A.capture_release()

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
        while 1:
            pyK4A.device_get_capture()
            depth_image_handle = pyK4A.capture_get_depth_image()
            color_image_handle = pyK4A.capture_get_color_image()
            if depth_image_handle and color_image_handle:
                break
        color_image = pyK4A.image_convert_to_numpy(color_image_handle)[ROI[0]:ROI[1], ROI[2]:ROI[3]]
        # depth_image = pyK4A.image_convert_to_numpy(depth_image_handle)
        transformed_depth_image = pyK4A.transform_depth_to_color(depth_image_handle, color_image_handle)
        # maximum_hole_size = 10
        # smoothed_depth_image = postProcessing.smooth_depth_image(transformed_depth_image, maximum_hole_size)
        #
        # transformed_depth_color_image = cv2.applyColorMap(np.round(smoothed_depth_image / 30).astype(np.uint8),
        #                                                   cv2.COLORMAP_JET)[ROI[0]:ROI[1], ROI[2]:ROI[3]]

        transformed_depth_color_image = cv2.applyColorMap(np.round(transformed_depth_image / 30).astype(np.uint8),
                                                          cv2.COLORMAP_JET)[ROI[0]:ROI[1], ROI[2]:ROI[3]]
        pyK4A.image_release(depth_image_handle)
        pyK4A.image_release(color_image_handle)

        cv2.imwrite(root + '{}_0_Color.jpg'.format(FileTime), color_image)
        end1 = time.time()
        CamTime = round(end1 - start1, 2)

        start2 = time.time()
        # yolocmd.wb_yolo()
        img = cv2.imread(root + '{}_0_Color.jpg'.format(FileTime))
        wbmid, water = pyYolo.main(img, FileTime)
        end2 = time.time()
        YoloTime = round(end2 - start2, 2)

        start3 = time.time()
        num_wbmid, num_water, box = sortlist.wb_sort(wbmid, water)
        if num_wbmid != num_water:
            cv2.imwrite(root + "img_fix/" + '{}_0_Color.jpg'.format(FileTime), color_image)
        if box ==0:
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
                box_z = int(transformed_depth_image[i[1] + HW[0]//2 - ROIcut[0]][i[0] + HW[1]//2 - ROIcut[1]])
                box_xyz.append(i[:]+[box_z])
            ratio_list=[]
            # for i in box_xyz:
            #     print(65 / math.hypot(i[0] - i[2], i[1] - i[3]))
            #     ratio_list.append(65 / math.hypot(i[0] - i[2], i[1] - i[3]))
            # ratio_av = sum(ratio_list) / len(ratio_list)


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

            # xu, xd, yu, yd = 0, 0, 0, 0
            # while 1:
            #     scan = transformed_depth_image[target[1]+HW[0]//2-ROIcut[0] + xu][target[0]+HW[1]//2-ROIcut[1]]
            #     if scan == 0 or (scan - target[5])**2 > 90000:
            #         # print(scan)
            #         # print(xu)
            #         break
            #     xu = xu + 1
            #
            # while 1:
            #     scan = transformed_depth_image[target[1]+HW[0]//2-ROIcut[0] - xd][target[0]+HW[1]//2-ROIcut[1]]
            #     if scan == 0 or (scan - target[5])**2 > 90000:
            #         # print(scan)
            #         # print(xd)
            #         break
            #     xd = xd + 1
            #
            # while 1:
            #     scan = transformed_depth_image[target[1]+HW[0]//2-ROIcut[0]][target[0]+HW[1]//2-ROIcut[1] + yu]
            #     if scan == 0 or (scan - target[5])**2 > 90000:
            #         # print(scan)
            #         # print(yu)
            #         break
            #     yu = yu + 1
            #
            # while 1:
            #     scan = transformed_depth_image[target[1]+HW[0]//2-ROIcut[0]][target[0]+HW[1]//2-ROIcut[1] - yd]
            #     if scan == 0 or (scan - target[5])**2 > 90000:
            #         # print(scan)
            #         # print(yd)
            #         break
            #     yd = yd + 1
            #
            # print("xu, xd, yu, yd: ", xu, xd, yu, yd)
            # print(transformed_depth_image[511+HW[0]//2-ROIcut[0]][596+HW[1]//2-ROIcut[1]])
            # cv2.circle(color_image, (596, 511), 6, (0, 255, 0), 1)
            # targetFix = [target[0] + (yu - yd) // 2, target[1] + (xu - xd) // 2] + target[2:]
            # print("fix target:", targetFix)
            # cv2.circle(color_image, (target[0], target[1]), 6, (255, 0, 0), 1)
            # cv2.imwrite(root + '{}_3_Tarfix.jpg'.format(FileTime), color_image)

            # ratio = 65 / math.hypot(target[0] - target[2], target[1] - target[3])
            # print("ratio: ", ratio)
            # print("ratio_av:", ratio_av)

            cv2.circle(color_image, (target[0], target[1]), 6, (0, 0, 255), 2)
            cv2.rectangle(color_image, (70, 0), (835, 639), (0, 0, 255), 1)
            cv2.imwrite(root + '{}_3_Tar.jpg'.format(FileTime), color_image)

            cv2.imwrite(root + '{}_2_Depth.jpg'.format(FileTime), transformed_depth_color_image)

            # Focal length of ToF sensor is ~1.8 mm and Focal length of RGB sensor is ~2.3 mm
            if 1750 < target[5] < 1950:  # 1873
                ratio = 1.985
            elif 2050 < target[5] < 2250:  # 2189
                ratio = 2.317
            elif 2350 < target[5] < 2550:  # 2476
                ratio = 2.655
            elif 2650 < target[5] < 2850:  # 2775
                ratio = 2.992
            else:
                return "ratio is none"
            target = [int((target[0]-ROIcut[1])*ratio),
                      -int((target[1]-ROIcut[0])*ratio),
                      target[4],
                      target[5]]

            f = open(root + '{}_4_Robot.txt'.format(FileTime), 'w')
            f.write(str(target[0]) + " ")
            f.write(str(target[1]) + " ")
            f.write(str(target[2]) + " ")
            f.write(str(target[3]) + " ")
            f.write(str(num_wbmid))
            f.close()
            print("Robot target: ", target)

        end3 = time.time()
        SortTime = round(end3 - start3, 2)

        f = open(root + '{}_5_Time.txt'.format(FileTime), 'w')
        f.write("CamTime: " + str(CamTime) + "\n")
        f.write("YoloTime: " + str(YoloTime) + "\n")
        f.write("SortTime: " + str(SortTime) + "\n")
        f.close()
        pyK4A.capture_release()
        totaltimeend = time.time()
        print("CamTime:  ", CamTime)
        print("YoloTime: ", YoloTime)
        print("SortTime: ", SortTime)
        print("TotalTime:", round(totaltimeend - totaltime, 2))
        # break
        f = open('/home/rs_waterbox/CameraCMD.txt', 'w')
        f.write("2")
        f.close()


    pyK4A.device_stop_cameras()
    pyK4A.device_close()
    # c = 86
    # k = 0
    # pyK4A.device_open()
    # device_config = pyK4A.config
    # device_config.color_format = _k4a.K4A_IMAGE_FORMAT_COLOR_BGRA32
    # device_config.color_resolution = _k4a.K4A_COLOR_RESOLUTION_1080P
    # device_config.depth_mode = _k4a.K4A_DEPTH_MODE_WFOV_2X2BINNED
    # device_config.calibration_type = _k4a.K4A_CALIBRATION_TYPE_COLOR
    # print(device_config)
    #
    # # Start cameras using modified configuration
    # pyK4A.device_start_cameras(device_config)
    # HW = [1080, 1920]
    # ROIcut = [320, 440]
    # ROI = [HW[0] // 2 - ROIcut[0], HW[0] // 2 + ROIcut[0], HW[1] // 2 - ROIcut[1], HW[1] // 2 + ROIcut[1]]  # [y,x]
    # # Get capture
    # while True:
    #
    #     pyK4A.device_get_capture()
    #     # Get the depth image from the capture
    #     depth_image_handle = pyK4A.capture_get_depth_image()
    #     # Get the color image from the capture
    #     color_image_handle = pyK4A.capture_get_color_image()
    #     # Check the image has been read correctly
    #     if depth_image_handle and color_image_handle:
    #         # Read and convert the image data to numpy array:
    #         color_image = pyK4A.image_convert_to_numpy(color_image_handle)[ROI[0]-50:ROI[1]+50, ROI[2]-50:ROI[3]+50]
    #         # depth_image = pyK4A.image_convert_to_numpy(depth_image_handle)
    #
    #         # Transform the depth image to the color format
    #         transformed_depth_image = pyK4A.transform_depth_to_color(depth_image_handle, color_image_handle)
    #         # pyK4A.transformation_depth_image_to_color_camera
    #         # maximum_hole_size = 10
    #         # smoothed_depth_image = postProcessing.smooth_depth_image(transformed_depth_image, maximum_hole_size)
    #
    #         # Convert depth image (mm) to color, the range needs to be reduced down to the range (0,255)
    #         # transformed_depth_color_image = cv2.applyColorMap(np.round(smoothed_depth_image / 30).astype(np.uint8),
    #         #                                                   cv2.COLORMAP_JET)
    #
    #         # Add the depth image over the color image:
    #         # combined_image = cv2.addWeighted(color_image, 0.7, transformed_depth_color_image, 0.3, 0)
    #
    #         # Plot the image
    #         cv2.circle(color_image, (int(color_image.shape[1])//2, int(color_image.shape[0])//2), 12, (0, 0, 255), 2)
    #         cv2.namedWindow('Colorized Depth Image', cv2.WINDOW_NORMAL)  # -46.765  1623.719  -121.743
    #         cv2.imshow('Colorized Depth Image', color_image)
    #         k = cv2.waitKey(25)
    #
    #     if k == 27:  # Esc key to stop
    #         # cv2.imwrite("KinectDK_depth_color_image.jpg", transformed_depth_color_image)
    #         print(color_image.shape)
    #         # print(smoothed_depth_image.shape)
    #         # print(smoothed_depth_image[540][960])
    #         # print(smoothed_depth_image[540][1060])
    #         pyK4A.image_release(depth_image_handle)
    #         pyK4A.image_release(color_image_handle)
    #         pyK4A.capture_release()
    #         break
    #     elif k == 32 or k == 13:
    #         cv2.imwrite("/home/wb_kinect/" + "wb_kinect_{}.jpg".format(c), color_image)
    #         print("saved wb_kinect_{}.jpg".format(c))
    #         c = c+1
    # pyK4A.device_stop_cameras()
    # pyK4A.device_close()


if __name__ == "__main__":
    # start = time.time()
    main()
    # end = time.time()
    # print("Time:", round(end - start, 2))
