#!/usr/bin/env python
# coding=utf-8

import os
import numpy as np
import cv2

USERNAME = os.getenv('USERNAME')

# 在OCR识别发票的时候，有时候是不需要识别公章内容，这个时候，可以先把公章去掉：

#去章处理方法
def remove_stamp(input_file, save_file):
    # img = cv2.imread(input_file,cv2.IMREAD_COLOR)
    img = cv2.imdecode(np.fromfile(input_file, dtype=np.uint8), 1)  # 读取中文路径
    B_channel,G_channel,R_channel=cv2.split(img)     # 注意cv2.split()返回通道顺序
    _,RedThresh = cv2.threshold(R_channel,170, 355,cv2.THRESH_BINARY)
    # cv2.imwrite(save_file,RedThresh)
    cv2.imencode(os.path.splitext(save_file)[-1], RedThresh)[1].tofile(save_file) # 保存到中文路径


def main():
    input_file = rf"D:\Users\{USERNAME}\data\协查通知元素识别\image\1.png"
    save_file = rf"D:\Users\{USERNAME}\data\协查通知元素识别\image\1_去章.png"
    remove_stamp(input_file, save_file)


if __name__ == "__main__":
    main()


# 资料来源：https://github.com/guanshuicheng/invoice/app.py
