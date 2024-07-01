import os
from PIL import Image, ImageDraw
import xml.etree.ElementTree as ET
import cv2
import numpy as np
# # 设置输入文件夹和输出文件夹的路径
# input_folder = 'crop-full'
# output_folder = 'path/to/output/folder'

import numpy as np
import math
from collections import deque
import cv2
import numpy as np
def cross_product(x1, y1, x2, y2):
    return x1 * y2 - x2 * y1

def on_segment(x1, y1, x2, y2, x3, y3):
    return min(x1, x2) <= x3 <= max(x1, x2) and min(y1, y2) <= y3 <= max(y1, y2)

def line_intersect(x1, y1, x2, y2, x3, y3, x4, y4):
    d1 = cross_product(x3 - x1, y3 - y1, x2 - x1, y2 - y1)
    d2 = cross_product(x4 - x1, y4 - y1, x2 - x1, y2 - y1)
    d3 = cross_product(x1 - x3, y1 - y3, x4 - x3, y4 - y3)
    d4 = cross_product(x2 - x3, y2 - y3, x4 - x3, y4 - y3)

    if d1 * d2 < 0 and d3 * d4 < 0:
        return True
    if d1 == 0 and on_segment(x1, y1, x2, y2, x3, y3):
        return True
    if d2 == 0 and on_segment(x1, y1, x2, y2, x4, y4):
        return True
    if d3 == 0 and on_segment(x3, y3, x4, y4, x1, y1):
        return True
    if d4 == 0 and on_segment(x3, y3, x4, y4, x2, y2):
        return True

    return False

# def line_intersect(x1, y1, x2, y2, x3, y3, x4, y4):
#     """
#     判断线段 (x1,y1)-(x2,y2) 和线段 (x3,y3)-(x4,y4) 是否相交
#     返回 True 表示相交，False 表示不相交
#     """
#
#     def cross_product(x1, y1, x2, y2):
#         """ 计算向量 (x1,y1) 和向量 (x2,y2) 的叉积 """
#         return x1 * y2 - x2 * y1
#
#     # 判断两条线段所在的直线是否平行
#     if (y2 - y1) * (x4 - x3) == (y4 - y3) * (x2 - x1):
#         # 如果平行，判断两条直线是否重合（即在同一条直线上）
#         if cross_product(x2 - x1, y2 - y1, x4 - x3, y4 - y3) == 0:
#             # 在同一条直线上，判断是否有重合部分
#             if (min(x1, x2) <= x3 <= max(x1, x2) or
#                     min(x1, x2) <= x4 <= max(x1, x2) or
#                     min(x3, x4) <= x1 <= max(x3, x4) or
#                     min(x3, x4) <= x2 <= max(x3, x4)):
#                 return True
#         # 不在同一条直线上，肯定不相交
#         return False
#
#     # 计算两条直线的交点坐标
#     cross_x = ((x4 - x3) * (x1 * y2 - y1 * x2) - (x2 - x1) * (x3 * y4 - y3 * x4)) / (
#                 (x4 - x3) * (y2 - y1) - (x2 - x1) * (y4 - y3))
#     cross_y = ((y3 - y4) * cross_x + x3 * y4 - x4 * y3) / (x4 - x3)
#
#     # 判断交点是否在两条线段的内部
#     if (min(x1, x2) <= cross_x <= max(x1, x2) and
#             min(x3, x4) <= cross_x <= max(x3, x4) and
#             min(y1, y2) <= cross_y <= max(y1, y2) and
#             min(y3, y4) <= cross_y <= max(y3, y4)):
#         return True
#     return False


def find_nearest_black_pixel_bfs(binary_image, x,y):
    height, width = binary_image.shape
    visited = np.zeros((height, width), dtype=bool)
    queue = deque([(y, x, 0)])

    while queue:
        cur_x, cur_y, distance = queue.popleft()

        if not (0 <= cur_x < height and 0 <= cur_y < width) or visited[cur_x][cur_y]:
            continue

        visited[cur_x][cur_y] = True
        # cv2.circle(image, (cur_y, cur_x), 5, (0, 0, 255), -1)

        if binary_image[cur_x][cur_y] == 0:  # 黑色像素
            visited_image = np.zeros((height, width, 1), dtype=np.uint8)
            visited_image[visited] = 255
            # test[visited == True] = 100
            return (cur_y, cur_x)

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            next_x, next_y = cur_x + dx, cur_y + dy
            queue.append((next_x, next_y, distance + 1))
def pre_img(img):
    height, width, channels = img.shape

    # 计算缩放比例
    scale = min(512 / width, 512 / height)

    # 计算新的大小
    new_width = int(width * scale)
    new_height = int(height * scale)

    # 调整图片大小
    img = cv2.resize(img, (new_width, new_height))
    draw = img.copy()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((5, 5), np.uint8)
    dilation = cv2.dilate(gray, kernel, iterations=1)

    # 进行腐蚀操作
    erosion = cv2.erode(gray, kernel, iterations=1)
    mean = cv2.blur(dilation, (3, 3))
    mean2 = cv2.blur(dilation, (9, 9))

    # 进行膨胀操作

    # mean = cv2.medianBlur(gray, 3)
    # mean2 = cv2.medianBlur(gray, 9)\\\
    mean3 = cv2.blur(gray, (3, 3))
    mean4 = cv2.blur(gray, (9, 9))
    test = gray.copy()
    n = 0.93
    test[mean3 / mean4 > n] = 255
    test[mean3 / mean4 < n] = 0


    n = 0.93

    test=cv2.resize(test,(width,height))

    return test

def get_out_point_opencv(image,center_x,center_y,polygons,pointer_x,pointer_y):
    height, width, channels = image.shape
    scale = min(512 / width, 512 / height)

    # 计算新的大小
    new_width = int(width * scale)
    new_height = int(height * scale)

    image = cv2.resize(image, (new_width, new_height))
    cv2.imwrite("1.jpg",image)
    test = pre_img(image)
    cv2.imwrite("2.jpg",test)
    # visT=test.copy()
    # 解析XML文件并获取关键点坐标
    # xml_path = os.path.join(input_folder, os.path.splitext(filename)[0] + '.xml')
    # tree = ET.parse(xml_path)
    # root = tree.getroot()
    # center_x, center_y = root.find("./object[name='center']/points/points").text.split(',')
    # pointer_x, pointer_y = root.find("./object[name='pointer_tip']/points/points").text.split(',')
    # for obj in root.iter('object'):
    #     if str(obj.find('name').text) == 'gauge':
    #         bndbox = obj.find('bndbox')
    #         gxmin = float(bndbox.find('xmin').text)
    #         gymin=float(bndbox.find('ymin').text)
    center_x =float(center_x)* scale
    center_y =float(center_y)* scale
    pointer_x=float(pointer_x)* scale
    pointer_y=float(pointer_y)* scale
    polygons = [array * scale for array in polygons]


    points=[]
    for pts in polygons:

        mask = np.zeros_like(test)#将矩形框涂白
        pts = pts.astype(int)
        # pts = np.array(rotated_points, np.int32)
        cv2.fillPoly(mask, [pts], 255)
        test[mask == 255] = 255

        distances = np.sqrt((pts[:, 0] - center_x) ** 2 + (pts[:, 1] - center_y) ** 2)
        max_index = np.argsort(distances)
        max_points = pts[max_index]
        # distances = distances[max_index]
        # # print('距离最大的两个点为：', max_points)
        # max_distance = distances[max_index[2]]
        # less_max = distances[max_index[0]]
        # print('距离最大的两个点为：', max_points, '距离为：', less_max,max_distance)
        binary_image = np.array(test)
        rcx = (pts[0][0] + pts[1][0] + pts[2][0] + pts[3][0]) // 4
        rcy = (pts[0][1] + pts[1][1] + pts[2][1] + pts[3][1]) // 4
        x = rcx
        y = rcy

        # cv2.line(image, (max_points[0][0],max_points[0][1]), (max_points[1][0],max_points[1][1]), (0, 0, 255), 3)
        # cv2.line(image, (rcx,rcy), (int(center_x),int(center_y)), (0, 0, 255),3)
        # cv2.circle(image, (int(pointer_x), int(pointer_y)), 25, (255, 255, 0), -1)
        # len_pointer=math.sqrt((center_x - pointer_x)**2 + (center_y - pointer_y)**2)
        # dis_det=math.sqrt((center_x - rcx)**2 + (center_y - rcy)**2)
        # print(dis_det,len_pointer*0.9)
        if line_intersect(max_points[0][0],max_points[0][1],max_points[1][0],max_points[1][1],rcx,rcy,center_x,center_y):
            x = (max_points[0][0] + max_points[1][0]) // 2
            y = (max_points[0][1] + max_points[1][1]) // 2
        else:

            x = (max_points[0][0] + max_points[2][0]) // 2
            y = (max_points[0][1] + max_points[2][1]) // 2
            # print("22@")
        # x, y = 0, 0
        result = find_nearest_black_pixel_bfs(test, x, y)
        if result is not None:
            px, py = result
        # cv2.circle(test2, (x, y), 5, (135, 206, 255), -1)
        # print("cx,cy",center_x,center_y)
        # cv2.circle(image, (int(center_x), int(center_y)), 50, (255, 255, 0), -1)
        # cv2.circle(image, (px, py), 5, (0, 255, 0), -1)
        # cv2.circle(image, (x, y), 5, (255, 0, 0), -1)
            points.append((px/scale,py/scale))
        # return 
    return points



