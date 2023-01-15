#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 图片4通道转3通道 (转换RBGA图片成RGB)
# RGBA是代表Red（红色）Green（绿色）Blue（蓝色）和Alpha的色彩空间。虽然它有的时候被描述为一个颜色空间，但是它其实仅仅是RGB模型的附加了额外的信息。
# alpha通道一般用作不透明度参数。如果一个像素的alpha通道数值为0%，那它就是完全透明的（也就是看不见的），而数值为100%则意味着一个完全不透明的像素（传统的数字图像）。在0%和100%之间的值则使得像素可以透过背景显示出来，就像透过玻璃（半透明性），这种效果是简单的二元透明性（透明或不透明）做不到的。它使数码合成变得容易。alpha通道值可以用百分比、整数或者像RGB参数那样用0到1的实数表示。
# 有时它也被写成ARGB（像RGBA一样，但是第一个数据是alpha）。比如，0x80FFFF00是50%透明的黄色，因为所有的参数都在0到255的范围内表示。0x80是128，大约是255的一半。
# PNG是一种使用RGBA的图像格式。

# 获取python中颜色对应的RGB数字
# 方法1
from matplotlib import colors
orange_rgb = colors.hex2color(colors.cnames['yellow'])
print(orange_rgb)
# (1.0, 1.0, 0.0)

# 方法2：
from matplotlib import colors
print(colors.to_rgba('yellow')) # -> (1.0, 1.0, 0.0, 1.0)
print(colors.to_rgb('yellow')) # -> (1.0, 1.0, 0.0)

# 方法3：
from PIL  import ImageColor
print(ImageColor.getcolor('yellow','RGBA')) # (255, 255, 0, 255)
print(ImageColor.getcolor('yellow','RGB')) # (255, 255, 0)

# 图像的 alpha 通道.
# (1) Alpha通道,范围0-1,用来存储这个像素点的透明度
# (2 ) 真正让图片变透明的不是Alpha 实际是 Alpha 所代表的数值和其他数值做了一次运算
# (3) Alpha 通道使用8位二进制数，就可以表示256级灰度，即256级的透明度。白色（值为255）的 Alpha 像素用以定义不透明的彩色像素，而黑色（值为0）的 Alpha 通道像素用以定义透明像素，介于黑白之间的灰度（值为0-255）的 Alpha 像素用以定义不同程度的半透明像素。
# (4) 将图片 a 绘制到另一幅图片 b 上，如果图片 a 没有 alpha 通道，那么就会完全将 b 图片的像素给替换掉。而如果有 alpha 通道，那么最后覆盖的结果值将是 c = a * alpha + b * (1 - alpha)

# 把 RGBA 图片抓换成 RGB 图片,
# 方法1：PIL 的
from PIL import Image
Image.open(img_path).convert('RGBA')

img = Image.open(jpeg)
if img.mode != 'RGB':
    img = img.convert('RGB')
    print(jpeg)
    os.remove(jpeg)
    img.save(jpeg)

# 方法2：opencv使用
import cv2
cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)

# 方法3：自定义转换
import cv2
import numpy as np
import matplotlib.pyplot as plt


def rgba2rgb(rgba, background=(255, 255, 255)):
    row, col, ch = rgba.shape

    if ch == 3:
        return rgba

    assert ch == 4, 'RGBA image has 4 channels.'

    # 生成一个三维画布图片
    rgb = np.zeros((row, col, 3), dtype='float32')

    # 获取图片每个通道数据
    r, g, b, a = rgba[:, :, 0], rgba[:, :, 1], rgba[:, :, 2], rgba[:, :, 3]

    # 把 alpha 通道的值转换到 0-1 之间
    a = np.asarray(a, dtype='float32') / 255.0

    # 得到想要生成背景图片每个通道的值
    R, G, B = background

    # 将图片 a 绘制到另一幅图片 b 上，如果有 alpha 通道，那么最后覆盖的结果值将是 c = a * alpha + b * (1 - alpha)
    rgb[:, :, 0] = r * a + (1.0 - a) * R
    rgb[:, :, 1] = g * a + (1.0 - a) * G
    rgb[:, :, 2] = b * a + (1.0 - a) * B

    # 把最终数据类型转换成 uint8
    return np.asarray(rgb, dtype='uint8')


image_name_have_alpha = "huaxiao/tuxinghuaxiao/img/剪刀/0.png"

image_have_alpha = cv2.imread(image_name_have_alpha, cv2.IMREAD_UNCHANGED)
# 使用opencv自带的库把BGRA图片转换成BGR图片（注意，opencv读取进去的图片格式是 BGR）
image_have_alpha_convert_bgr1 = cv2.cvtColor(image_have_alpha, cv2.COLOR_RGBA2BGR)
# 使用自己的函数把BGRA图片转换成BGR图片，背景色设置为白色
image_have_alpha_convert_bgr2 = rgba2rgb(image_have_alpha)
# 使用自己的函数把BGRA图片转换成BGR图片，背景色设置为蓝色
image_have_alpha_convert_bgr3 = rgba2rgb(image_have_alpha, background=(0, 0, 255))
plt.figure()
plt.subplot(2, 2, 1), plt.title("original image"), plt.imshow(image_have_alpha, "gray")
plt.subplot(2, 2, 2), plt.title("use function of cv2"), plt.imshow(image_have_alpha_convert_bgr1, "gray")
plt.subplot(2, 2, 3), plt.title("use function of myself"), plt.imshow(image_have_alpha_convert_bgr2, "gray")
plt.subplot(2, 2, 4), plt.title("use function of myself"), plt.imshow(image_have_alpha_convert_bgr3, "gray")
plt.show()

def main():
    pass


if __name__ == '__main__':
    main()
