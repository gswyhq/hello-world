#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import cv2
import numpy as np
import os, math
from PIL import Image

USERNAME = os.getenv("USERNAME")

img_path = rf"D:\Users\{USERNAME}\github_project\PJ_PREDICT_IMG\68747.png"
# out_path = rf"D:\Users\{USERNAME}\github_project\PJ_PREDICT_IMG\68747_cut"
image = cv2.imread(img_path)  # 读取图像

# 转换为灰度图
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# 转为二值图,二值化
# 在灰度图的基础上运用了adaptiveThreshold来达成自动阈值的二值化，这个算法在提取直线和文字的时候比Canny更好用，其中~gray的意思是反色，使二值化后的图片是黑底白字，才能正确的提取直线。
# ret, binary = cv2.threshold(gray, black_thr, 255, cv2.THRESH_BINARY)
# binary = cv2.adaptiveThreshold(~gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, -5)
binary = cv2.adaptiveThreshold(~gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

# 膨胀算法的色块大小
h, w = binary.shape
scale = 1.2 # 这个值越小，检测到的直线越多
hors_k = int(math.sqrt(w) * scale)
vert_k = int(math.sqrt(h) * scale)

# 先腐蚀再膨胀；
# 白底黑字，膨胀白色横向色块，抹去文字和竖线，保留横线
# 为了获取横向的表格线，设置腐蚀和膨胀的操作区域为一个比较大的横向直条
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (hors_k, 1))
hors = ~cv2.dilate(binary, kernel, iterations=2)  # 迭代两次，尽量抹去文本横线，变反为黑底白线

# 白底黑字，膨胀白色竖向色块，抹去文字和横线，保留竖线
# 竖直方向上线条获取的步骤同上，唯一的区别在于腐蚀膨胀的区域为一个宽为1，高为缩放后的图片高度的一个竖长形直条
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vert_k))
verts = ~cv2.dilate(binary, kernel, iterations=2)  # 迭代两次，尽量抹去文本竖线，变反为黑底白线

# 横线竖线检测结果合并
borders = cv2.bitwise_or(hors, verts)
cv2.imshow("borders", borders)

# cv2.waitKey(0)
#########################################################################################################################
# 若是标准表格的话，则客户交叉横纵线条，定位点
mask = hors + verts
cv2.imshow("mask", mask)

# 通过bitwise_and函数获得横纵线条的交点，通过交点再做后面的表格提取操作
joints = cv2.bitwise_and(hors, verts)
cv2.imshow("joints", joints)

# 判断区域是否为表格
# 在mask那张图上通过findContours 找到轮廓，判断轮廓形状和大小是否为表格。
# approxPolyDP 函数用来逼近区域成为一个形状，true值表示产生的区域为闭合区域。
# boundingRect 为将这片区域转化为矩形，此矩形包含输入的形状。返回四个值，分别是x，y，w，h； x，y是矩阵左上点的坐标，w，h是矩阵的宽和高
contours, HIERARCHY = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

bounding_boxes = [cv2.boundingRect(cnt) for cnt in contours]

for bbox in bounding_boxes:
    [x, y, w, h] = bbox
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # 利用cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)画出矩行
    # 参数解释
    # 第一个参数：img是原图
    # 第二个参数：（x，y）是矩阵的左上点坐标
    # 第三个参数：（x+w，y+h）是矩阵的右下点坐标
    # 第四个参数：（0,255,0）是画线对应的rgb颜色
    # 第五个参数：2是所画的线的宽度

cv2.imshow("name", image)
cv2.waitKey(0)

joints_contours = []
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area < 500:
        continue

    epsilon = 0.1 * cv2.arcLength(cnt, True)
    poly_cnt = cv2.approxPolyDP(cnt, epsilon, True) # 函数用来逼近区域成为一个形状，true值表示产生的区域为闭合区域。
    x,y,w,h = cv2.boundingRect(poly_cnt) # 获取最小外接矩形, 为将这片区域转化为矩形，此矩形包含输入的形状。返回四个值，分别是x，y，w，h； x，y是矩阵左上点的坐标，w，h是矩阵的宽和高
    # roi = image.copy()
    # cv2.rectangle(roi,(x,y),(x+w,y+h),(255,0,0),3)  # 参数表示依次为： （图片，长方形框左上角坐标, 长方形框右下角坐标， 字体颜色，字体粗细）
    # # cv2.rectangle 在图片img上画长方形，坐标原点是图片左上角，向右为x轴正方向，向下为y轴正方向。左上角（x，y），右下角（x，y） ，颜色(B,G,R), 线的粗细
    roi = joints[y:y+h, x:x+w] # 查找每个表的关节数
    joints_contour, _ = cv2.findContours(roi, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    # 从表格的特性看，如果这片区域的点数小于4，那就代表没有一个完整的表格，忽略掉
    joints_contours.extend([t for t in joints_contour if len(t)>=4])

draw_img0 = cv2.drawContours(image.copy(),joints_contours,0,(0,255,255),3)

# ————————————————
#
# 原文链接：https://blog.csdn.net/weixin_41189525/article/details/121889157
# https://blog.csdn.net/yomo127/article/details/52045146

#########################################################################################################################

def main():
    pass


if __name__ == '__main__':
    main()
