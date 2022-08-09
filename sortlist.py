# -*- coding: utf-8 -*-
from scipy import spatial
import math
import time

def wb_sort(label0, label1):
    root = "/home/sunny/wb_data/"

    # f = open('/home/sunny/darknet/wb/YoloList.txt', 'r')
    # FileTime = f.readlines()
    # FileTime = FileTime[0].split("/")[-1].split("_")
    # FileTime = FileTime[0] + "_" + FileTime[1]
    # print(FileTime)
    # f.close()

    # file = root + "{}_Pre.txt".format(FileTime)  # 0104_111858_Pre  "{}_Pre.txt"
    # # file = root + "0104_151654_Pre.txt"
    # # case1 = [[120, 110, 0], [118, 213, 0], [129, 320, 0], [113, 445, 1],
    # #          [254, 134, 1], [256, 301, 1], [221, 452, 1], [318, 454, 1],
    # #          [389, 104, 0], [390, 208, 0], [390, 314, 0], [434, 454, 1],
    # #          [544, 109, 0], [548, 220, 0], [533, 320, 0], [546, 462, 1]]
    # #
    # # case2 = [[109, 463, 0], [100, 356, 0], [102, 255, 0], [100, 121, 1],
    # #          [206, 117, 1], [262, 247, 0], [259, 354, 0], [261, 457, 0],
    # #          [396, 435, 1], [393, 278, 1], [302, 120, 1], [421, 116, 1],
    # #          [526, 131, 1], [524, 251, 0], [529, 361, 0], [512, 463, 0]]
    #
    # # wh = [700, 600]
    # # print("wblayout: ", wblayout)
    # f = open(file, "r")
    # wbmid = []
    # water = []
    # for i in f:
    #     if "wbmid" in i:
    #         if int(i.split("\t")[0].split(" ")[1].split("%")[0]) > 88:
    #             x = i.replace("  ", " ").replace("  ", " ").split(" ")[2]
    #             y = i.replace("  ", " ").replace("  ", " ").split(" ")[4]
    #             h = i.replace("  ", " ").replace("  ", " ").split(" ")[6]
    #             w = i.replace("  ", " ").replace("  ", " ").split(" ")[8].split(")")[0]
    #             wbmid.append([int(x) + int(int(w) * 0.5), int(y) + int(int(h) * 0.5)])
    #     elif "water" in i:
    #         if int(i.split("\t")[0].split(" ")[1].split("%")[0]) > 88:
    #             x = i.replace("  ", " ").replace("  ", " ").split(" ")[2]
    #             y = i.replace("  ", " ").replace("  ", " ").split(" ")[4]
    #             h = i.replace("  ", " ").replace("  ", " ").split(" ")[6]
    #             w = i.replace("  ", " ").replace("  ", " ").split(" ")[8].split(")")[0]
    #             water.append([int(x) + int(int(w) * 0.5), int(y) + int(int(h) * 0.5)])
    # print("wbmid:", wbmid)
    # print("water:", water)
    #
    # wbmid = [s for s in wbmid if 785 > s[0] > 120]
    # water = [s for s in water if 785 > s[0] > 120]
    wbmid = [s for s in label0 if 835 > s[0] > 70]
    water = [s for s in label1 if 835 > s[0] > 70]
    print("sort wbmid:", wbmid)
    print("sort water:", water)

    if len(wbmid) ==0 or len(water) ==0:
        return 0, None, 0
    else:
        tree = spatial.KDTree(water)
        # distance, index = tree.query([360, 284])  # [375, 153]   [383, 283]
        # print("test:", water[index])
        box = []
        for i in range(len(wbmid)):
            distance, index = tree.query(wbmid[i])
            if (wbmid[i][0] - water[index][0])**2 < (wbmid[i][1] - water[index][1])**2 :
                vec = 1
            else:
                vec = 0
            box.append([wbmid[i][0], wbmid[i][1], water[index][0], water[index][1], vec])

        # print(box)
        print("sort box: ", len(box))
        box = sorted(box, key=lambda m: (m[1]), reverse=1)
        # print(box)
        # list1 = []
        # for i in range(len(box)):
        #     if (box[0][1]-box[i][1]) ** 2 < 1000:
        #         list1.append(box[i])
        # print("list1:", list1)
        # list1 = sorted(list1, key=lambda m: (m[0]), reverse=1)
        # print("list1:", list1[0][0:2])

        # distance, index = tree.query(list1[0][0:2])
        # print("pair water:", water[index], index)

        # ratio_water_wbmid = 66/math.hypot(water[index][0] - list1[0][0], water[index][1] - list1[0][1])

        # f = open(root + '{}_Target.txt'.format(FileTime), 'w')
        # f.write(str(list1[0][0])+" ")
        # f.write(str(list1[0][1])+" ")
        # f.write(str(list1[0][2]))
        # f.close()

        # return ([list1[0][0], list1[0][1], list1[0][2]]), len(wbmid), ratio_water_wbmid, box
        return len(wbmid), len(water), box


if __name__ == "__main__":
    l1 = [[500, 194], [390, 458], [391, 333], [609, 194], [593, 329], [491, 331], [390, 206], [491, 456], [591, 456]]
    l2 = [[501, 220], [609, 218], [390, 354], [390, 479], [389, 228], [593, 308], [491, 352], [490, 476], [590, 478], [591, 478]]
    start = time.time()
    wbmidnum, waternumm, target = wb_sort(l1, l2)
    print("wbmidnum, waternumm, target:", wbmidnum, waternumm, target)
    end = time.time()
    # print("Time:", round(end - start, 6))
    # ratio = 123
    # target = [int((target[0][0] - 240) * ratio * 1000),
    #           int((target[0][1] - 320) * ratio * 1000),
    #           target[0][2]]
    # print("target:", target)

# print(sorted(wblayout, key=lambda m: (m[2]), reverse=1))
# savelist = sorted(wblayout, key=lambda m: m[2], reverse=1)


# wblayout = case1
# print('check: ', sorted(wblayout, key=lambda m: (m[1]), reverse=1))
# check = sorted(wblayout, key=lambda m: (m[1]), reverse=1)   # find near y
# if check[0][2] == 1:
#     print("case1")
# elif check[0][2] == 0:
#     print("case2")
#
# # case 1
# caselist = []
# for i in check:
#     if (i[1] - check[0][1]) ** 2 < 5000:
#         caselist.append(i)
# print("caselist:", caselist)
# caselist = sorted(caselist, key=lambda m: (m[0]), reverse=0)
# print("caselist:", caselist)

# case 2


# print("=========================================")
# print('check: ', sorted(wblayout, key=lambda m: ((m[0]) ** 2 + (m[1] - wh[1]) ** 2), reverse=0))
# corner = sorted(wblayout, key=lambda m: ((m[0]) ** 2 + (m[1] - wh[1]) ** 2), reverse=0)[0]
# # print(corner)
# # print(corner[1])
# list1 = []
# for i in wblayout:
#     if (i[1] - corner[1]) ** 2 < 5000:
#         list1.append(i)
# print("list1:", list1)
# list1 = sorted(list1, key=lambda m: (m[0]), reverse=0)
# print("list1:", list1)
#
# for e in list1:
#     if e in wblayout:
#         wblayout.remove(e)
# print("wblayout: ", wblayout)
# print("=========================================")
# corner = sorted(wblayout, key=lambda m: ((m[0]) ** 2 + (m[1] - wh[1]) ** 2), reverse=0)[0]
# list2 = []
# for i in wblayout:
#     if (i[0] - corner[0]) ** 2 < 5000:
#         list2.append(i)
# print("list2:", list2)
# list2 = sorted(list2, key=lambda m: (m[1]), reverse=1)
# print("list2:", list2)

# num = 0
# list1 = []
# list2 = []
# list3 = []
# list4 = []
# th = 60
# for i in range(len(savelist) - 1):
#     if savelist[i + 1][1] - savelist[i][1] < -th:
#         list1.append(savelist[i])
#         break
#     else:
#         list1.append(savelist[i])
#
# print(list1)
# # print(savelist)
# for e in list1:
#     if e in savelist:
#         savelist.remove(e)
# print(savelist)
# print("=========================================")
#
# for i in range(len(savelist) - 1):
#     if savelist[i + 1][0] - savelist[i][0] > th:
#         list2.append(savelist[i])
#         break
#     else:
#         list2.append(savelist[i])
#
# print(list2)
# # print(savelist)
# for e in list2:
#     if e in savelist:
#         savelist.remove(e)
# print(savelist)
#
# print("=========================================")
#
# for i in range(len(savelist) - 1):
#     if savelist[i + 1][0] - savelist[i][0] > th:
#         list3.append(savelist[i])
#         break
#     else:
#         list3.append(savelist[i])
#
# print(list3)
# # print(savelist)
# for e in list3:
#     if e in savelist:
#         savelist.remove(e)
# print(savelist)
