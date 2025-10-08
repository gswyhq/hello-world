#!/usr/bin/python3
# coding: utf-8

import os
import cv2
from datetime import datetime
from PIL import Image, ImageFilter, ImageOps

def dodge(a, b, alpha):
    return min(int(a * 255 / (256 - b * alpha)), 255)

def draw(img, blur=25, alpha=1.0):
    # 图片转换成灰色
    img1 = img.convert('L')
    img2 = img1.copy()
    img2 = ImageOps.invert(img2)
    # 模糊度
    for i in range(blur):
        img2 = img2.filter(ImageFilter.BLUR)
    width, height = img1.size
    for x in range(width):
        for y in range(height):
            a = img1.getpixel((x, y))
            b = img2.getpixel((x, y))
            img1.putpixel((x, y), dodge(a, b, alpha))
    img1.show()
    return img1


def cartoonise(picture_name):

    imgInput_FileName = picture_name
    file_path, file_ext = os.path.splitext(picture_name)
    imgOutput_FileName = file_path + '_{}'.format(datetime.now().strftime('%Y%m%d_%H%M%S')) + file_ext
    num_down = 2         #缩减像素采样的数目
    num_bilateral = 7    #定义双边滤波的数目
    img_rgb = cv2.imread(imgInput_FileName)     #读取图片
    #用高斯金字塔降低取样
    img_color = img_rgb
    for _ in range(num_down):
        img_color = cv2.pyrDown(img_color)
    #重复使用小的双边滤波代替一个大的滤波
    for _ in range(num_bilateral):
        img_color = cv2.bilateralFilter(img_color,d=9,sigmaColor=9,sigmaSpace=7)
    #升采样图片到原始大小
    for _ in range(num_down):
        img_color = cv2.pyrUp(img_color)
    #转换为灰度并且使其产生中等的模糊
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    img_blur = cv2.medianBlur(img_gray, 7)
    #检测到边缘并且增强其效果
    img_edge = cv2.adaptiveThreshold(img_blur,255,
                                     cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY,
                                     blockSize=9,
                                     C=2)
    #转换回彩色图像
    img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2RGB)
    img_cartoon = cv2.bitwise_and(img_color, img_edge)
    # 保存转换后的图片
    cv2.imwrite(imgOutput_FileName, img_cartoon)
    print(imgOutput_FileName)




def main():
    jpg_path = r'/home/gswyhq/Downloads/000.jpg'
    img = Image.open(jpg_path)
    img1 = draw(img)
    img1.save(r'/home/gswyhq/Downloads/000_1.jpg')

    cartoonise(r'/home/gswyhq/Downloads/000.jpg')

if __name__ == '__main__':
    main()


'''
使用大模型将图片转换为简笔画(手绘图、线图)的方法：
使用模型 Qwen-Image-Edit
1、上传图片
2、输入提示词：Convert this image into a clean black and white line drawing with no colors or shading
或者中文提示词：将这张图片转换为清晰的黑白线描图，不包含任何色彩或阴影。

'''

