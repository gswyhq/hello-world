#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import os
from PIL import Image

USERNAME = os.getenv("USERNAME")


# 去章处理方法, 去除公章，去除红章
def remove_stamp(img=None, img_path=''):
    if img is None:
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    B_channel, G_channel, R_channel = cv2.split(img)  # 注意cv2.split()返回通道顺序; rgb, 172, 106, 64
    _, RedThresh = cv2.threshold(R_channel, 170, 355, cv2.THRESH_BINARY)

    # cv2.imwrite('./test/RedThresh_{}.jpg'.format(invoice_file_name), RedThresh)
    return RedThresh

def split_color(img=None, img_path=''):
    '''分离出指定颜色'''
    if img is None:
        img = cv2.imread(img_path)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # 色彩空间转换为hsv，便于分离
    # 不同颜色的范围可查 hsv色彩取值表
    # 这里是分离出褐色， rgb = (172, 106, 64)
    lowerb = np.array([11, 43, 46, ])
    upperb = np.array([34, 255, 255, ])
    mask = cv2.inRange(hsv, lowerb, upperb)
    # cv2.imwrite(rf"D:\Users\{USERNAME}\github_project\PJ_PREDICT_IMG\68747_6.png", mask)
    return mask

del_stamp=True
split_c = True
img_path = rf"D:\Users\{USERNAME}\github_project\PJ_PREDICT_IMG\68747.png"
img = cv2.imread(img_path)
if split_c:
    gray = split_color(img=img, img_path='')
elif del_stamp:
    gray = remove_stamp(img)
else:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  #转灰度图
gaus = cv2.GaussianBlur(gray, (3, 3), 0)  #高斯模糊
gaus = cv2.adaptiveThreshold(~gaus, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
edges = cv2.Canny(gaus, 50, 200, apertureSize=5)  # 边缘检测，低于阈值1的像素点会被认为不是边缘；高于阈值2的像素点会被认为是边缘；

def to_p_lines(lines):
    '''将返回值（ρ,θ），ρ以像素为单位，θ以弧度为单位。
    转换为：线的俩个端点。'''
    lines2 = []
    for i, line in enumerate(lines):
        for rho, theta in line:
            # print(rho, theta * 180 - 180)
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            lines2.append([[x1, y1, x2, y2]])
    return np.array(lines2)

# 霍夫变换(HoughLines)
# lines = cv2.HoughLines(edges, 1, np.pi / 180, 50, lines=None, srn=None, stn=None, min_theta=None, max_theta=None)  # 返回值（ρ,θ），ρ以像素为单位，θ以弧度为单位。
# 阈值，假设检测线，表示被检测到的线的最短长度，低于200像素长度的将被丢弃；
# lines = to_p_lines(lines)

# 概率霍夫变换(HoughLinesP)
# # 概率霍夫变换（Probablistic Hough Transform）是霍夫变换（Hough Transform）的优化。
# # 霍夫变换拿线的所有点参与计算，而概率霍夫变换不考虑所有的点，而是只考虑点的一个随机子集，这对于线检测是足够的。
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, lines=None, minLineLength=180, maxLineGap=9)  # 它直接返回线的俩个端点。
# HoughLinesP(image, rho, theta, threshold, lines=None, minLineLength=None, maxLineGap=None)
# image： 必须是二值图像，推荐使用canny边缘检测的结果图像；
# rho: 线段以像素为单位的距离精度，double类型的，推荐用1.0
# theta： 线段以弧度为单位的角度精度，推荐用numpy.pi/180
# threshod: 累加平面的阈值参数，int类型，超过设定阈值才被检测出线段，值越大，基本上意味着检出的线段越长，检出的线段个数越少。根据情况推荐先用100试试
# lines：这个参数的意义未知，发现不同的lines对结果没影响，但是不要忽略了它的存在
# minLineLength：线段以像素为单位的最小长度，小于此长度的线段将被丢弃；
# maxLineGap：同一方向上两条线段判定为一条线段的最大允许间隔（断裂），超过了设定值，则把两条线段当成一条线段，值越大，允许线段上的断裂越大，越有可能检出潜在的直线段


print(f"共检测出线条数：{len(lines)}")
lines = [line for line in lines if (line[0][0] == line[0][2]) or (line[0][1] == line[0][3])]

for idx, line in enumerate(lines):
    x1, y1, x2, y2 = line[0]
    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img, f'L({idx})', (x1+5, y1 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

cv2.imwrite(rf"D:\Users\{USERNAME}\github_project\PJ_PREDICT_IMG\68747_6.png", img)

###############################################################################################################################
# 基于检测的直线，提取方框
img = cv2.imread(img_path)
img2 = np.zeros(img.shape ) + 255
print(f"共检测出线条数：{len(lines)}")
for idx, line in enumerate(lines):
    x1, y1, x2, y2 = line[0]
    cv2.line(img2, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # cv2.putText(img2, f'L({idx})', (x1+5, y1 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
line_file = rf"D:\Users\{USERNAME}\github_project\PJ_PREDICT_IMG\68747_line.png"
cv2.imwrite(line_file, img2)

img2 = cv2.imread(line_file)
gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)  #转灰度图
binary = cv2.adaptiveThreshold(~gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
contours, _hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
print("轮廓数量：%d" % len(contours))


def angle_cos(p0, p1, p2):
    '''计算两边夹角额cos值'''
    d1, d2 = (p0 - p1).astype('float'), (p2 - p1).astype('float')
    return abs(np.dot(d1, d2) / np.sqrt(np.dot(d1, d1) * np.dot(d2, d2)))


index = 0
squares = []

# 轮廓遍历
for cnt in contours:
    cnt_len = cv2.arcLength(cnt, True)  # 计算轮廓周长
    cnt = cv2.approxPolyDP(cnt, 0.02 * cnt_len, True)  # 多边形逼近
    # 条件判断逼近边的数量是否为4，轮廓面积是否大于1000，检测轮廓是否为凸的
    # 通过cv2.contourArea可以求得轮廓面积
    # isContourConvex(contours)是用来判断某个轮廓是不是一个凸轮廓，其返回的是一个布尔值，当输入轮廓是凸轮廓时返回true，不是凸轮廓时返回false。
    if len(cnt) == 4 and cv2.contourArea(cnt) > 1000 and cv2.isContourConvex(cnt):
        M = cv2.moments(cnt)  # 计算轮廓的矩
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])  # 轮廓重心
        cnt = cnt.reshape(-1, 2)
        max_cos = np.max([angle_cos(cnt[i], cnt[(i + 1) % 4], cnt[(i + 2) % 4]) for i in range(4)])
        # 只检测矩形（cos90° = 0）
        if max_cos < 0.1:
            # 检测四边形（不限定角度范围）
            # if True:
            index = index + 1
            cv2.putText(img, ("#%d" % index), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            squares.append(cnt)

cv2.drawContours(img, squares, -1, (0, 0, 255), 2)
cv2.imwrite(rf"D:\Users\{USERNAME}\github_project\PJ_PREDICT_IMG\68747_6.png", img)

##########################################################################################################################
# 有时候提取效果并不规整，可以进一步获取最小外接矩形
squares2= []
for c in squares:  # 遍历轮廓
    rect = cv2.minAreaRect(c)  # 生成最小外接矩形
    box = cv2.boxPoints(rect)  # 计算最小面积矩形的坐标
    box = np.int0(box)  # 将坐标规范化为整数
    if cv2.contourArea(box)<500:
        continue
    squares2.append(box)

img = cv2.imread(img_path)
cv2.drawContours(img, squares2, -1, (0, 0, 255), 2)
cv2.imwrite(rf"D:\Users\{USERNAME}\github_project\PJ_PREDICT_IMG\68747_6.png", img)

def main():
    pass


if __name__ == '__main__':
    main()
