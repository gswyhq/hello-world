#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import cv2 
import numpy as np


def separate_color(frame):
    cv2.imshow("原图", frame)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # 色彩空间转换为hsv，便于分离
    # 不同颜色的范围可查 hsv色彩取值表
    lower_hsv = np.array([37, 43, 46])  # 提取颜色的低值
    high_hsv = np.array([77, 255, 255])  # 提取颜色的高值
    mask = cv2.inRange(hsv, lowerb=lower_hsv, upperb=high_hsv)  #
    # inRange(InputArray src, InputArray lowerb, InputArray upperb, OutputArray dst)
    # 参数介绍：src：输入图像， lowerb：要提取颜色在hsv色彩空间取值的低值， upperb：要提取颜色在hsv色彩空间取值的高值， 输出图
    cv2.imshow("inRange", mask)


image = "D:/Image/test1.jpg"
src = cv2.imread(image)
separate_color(src)

cv2.waitKey(0)

def main():
    pass


if __name__ == '__main__':
    main()
