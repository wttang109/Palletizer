import pyYolact
# import sortlist
import time
import math

import pyrealsense2 as rs
import numpy as np
import cv2
import json


# hardware_reset()   # https://github.com/IntelRealSense/librealsense/issues/5913
def main():
    root = "/home/sunny/wb_data/"
    rs_config = "/home/sunny/librealsense/default_no_post.json"
    jsonObj = json.load(open(rs_config))
    json_string = str(jsonObj).replace("'", '\"')

    reso = [int(jsonObj['stream-width']), int(jsonObj['stream-height'])]
    ColDep = [10, -6]
    # reso = [reso[0] + ColDep[0], reso[1] + ColDep[1]]

    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.depth, int(jsonObj['stream-width']), int(jsonObj['stream-height']),
                         rs.format.z16,
                         int(jsonObj['stream-fps']))
    config.enable_stream(rs.stream.color, int(jsonObj['stream-width']), int(jsonObj['stream-height']),
                         rs.format.bgr8,
                         int(jsonObj['stream-fps']))
    while True:

        try:
            cfg = pipeline.start(config)
            dev = cfg.get_device()
            advnc_mode = rs.rs400_advanced_mode(dev)
            advnc_mode.load_json(json_string)
            print("Open successful")
            break
        except:
            print("Reset Cam")
            time.sleep(1)
            # ctx = rs.context()
            # devices = ctx.query_devices()
            # for dev in devices:
            #     dev.hardware_reset()

    while True:
        start = time.time()
        FileTime = time.strftime("%m%d_%H%M%S", time.localtime())
        print(FileTime)
        while 1:
            f = open('/home/sunny/rs_waterbox/CameraCMD.txt', 'r')
            CameraCMD = f.readlines()[0].split('\n')[0]
            if CameraCMD == "DEPTH":
                f.close()
                break
            time.sleep(0.2)
            print("wait for robot move")

        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame and depth_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        # color_image = color_image[:, :960]
        cv2.imwrite(root + '{}_0_Color.jpg'.format(FileTime), color_image)

        box = pyYolact.yolact_main(FileTime)
        print("box_center:", box)

        colorizer = rs.colorizer()
        depth_image = np.asanyarray(colorizer.colorize(depth_frame).get_data())
        # print("reso", int(reso[0] / 2), int(reso[1] / 2))
        # print("target", target[0], target[1])
        box_xyz = []
        for i in box:
            box_z = int(depth_frame.get_distance(i[0], i[1]))
            box_xyz.append(i[:] + [box_z])

        box_xyz = sorted(box_xyz, key=lambda m: (m[3]), reverse=0)
        box_xyz = [s for s in box_xyz if (box_xyz[0][3] - s[3]) ** 2 < 0.0025]
        box_xyz = sorted(box_xyz, key=lambda m: (m[1]), reverse=1)
        box_xyz = [s for s in box_xyz if (box_xyz[0][1] - s[1]) ** 2 < 100]
        box_xyz = sorted(box_xyz, key=lambda m: (m[0]), reverse=1)

        target = box_xyz[0]
        # print("target", target[0])

        udist = depth_frame.get_distance(int(reso[0] / 2), int(reso[1] / 2))
        vdist = depth_frame.get_distance(target[0], target[1])
        # print("udist", udist)
        # print("vdist", vdist)

        color_intrin = color_frame.profile.as_video_stream_profile().intrinsics

        point1 = rs.rs2_deproject_pixel_to_point(color_intrin, [int(reso[0] / 2), int(reso[1] / 2)], udist)
        point2 = rs.rs2_deproject_pixel_to_point(color_intrin, [target[0] + ColDep[0], target[1] + ColDep[1]],
                                                 vdist)
        # if 0.0 in point1 or 0.0 in point2:
        #     print("Stop and Reopen")
        #     pipeline.stop()
        #     ctx = rs.context()
        #     devices = ctx.query_devices()
        #     for dev in devices:
        #         dev.hardware_reset()
        #     print("Reopen done")
        #     time.sleep(1)
        #     continue
        # else:
        #     break

        # f = open('/home/sunny/rs_waterbox/CameraCMD.txt', 'w')
        # f.write("MOVE")
        # f.close()

        f = open('/home/sunny/rs_waterbox/Time.txt', 'w')
        seq = str(FileTime)
        f.write(seq)
        f.close()

        f = open(root + '{}_4_Robot.txt'.format(FileTime), 'w')
        f.write(str(int(round(point2[0], 3) * 1000)) + " ")
        f.write(str(int(round(-point2[1], 3) * 1000)) + " ")
        f.write(str(target[2]) + " ")
        f.write(str(int(round(point2[2], 3) * 1000)))
        f.write(str(len(box)))
        f.close()

        cv2.circle(depth_image, (int(reso[0] / 2), int(reso[1] / 2)), 0, (0, 0, 255), 1)
        cv2.circle(depth_image, (target[0] + ColDep[0], target[1] + ColDep[1]), 0, (255, 0, 180), 1)
        cv2.imwrite(root + '{}_Depth.jpg'.format(FileTime), depth_image)
        #
        cv2.circle(color_image, (int(reso[0] / 2), int(reso[1] / 2)), 0, (0, 0, 255), 1)
        cv2.circle(color_image, (target[0], target[1]), 1, (255, 0, 180), 5)
        cv2.imwrite(root + '{}_Color_Tar.jpg'.format(FileTime), color_image)
        print("point1", point1)
        print("point2", point2)
        # print("offset", point1[2] - point2[2])
        # print("reso ", int(reso[0]/2), int(reso[1]/2))
        # print("target ", target[0], target[1])
        # point2[1] = -point2[1]

        print(int(round(point2[0], 3) * 1000000), ",",
              int(round(-point2[1], 3) * 1000000), ",",
              int(round(point2[2], 3) * 1000000))
        end = time.time()
        print("Time:", round(end - start, 2))
        break

    pipeline.stop()


if __name__ == "__main__":
    main()
