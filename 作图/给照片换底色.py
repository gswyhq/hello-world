#!/usr/bin/python3
# coding: utf-8
import cv2
import  numpy as np

# 图像载入
# 导入opencv库，使用imread函数读取图片

img=cv2.imread('/home/gswyhq/Downloads/微信图片_20190602192820.jpg')

# 由于证件照太大，不方便显示，故进行缩放（可有可无）

#缩放
rows,cols,channels = img.shape
img=cv2.resize(img,None,fx=0.5,fy=0.5)
rows,cols,channels = img.shape
cv2.imshow('img',img)

# 获取背景区域
# 首先将读取的图像默认BGR格式转换为HSV格式，然后通过inRange函数获取背景的mask。
# HSV颜色范围参数可调节根据这篇博客(https://blog.csdn.net/taily_duan/article/details/51506776)

# 蓝底 换 红底

hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
# lower_blue=np.array([78,43,46])
# upper_blue=np.array([110,255,255])
lower_blue=np.array([20,20,20])
upper_blue=np.array([200,200,200])
mask = cv2.inRange(hsv, lower_blue, upper_blue)
# 利用cv2.inRange函数设阈值，去除背景部分
# 函数很简单，参数有三个
# 第一个参数：hsv指的是原图
# 第二个参数：lower_red指的是图像中低于这个lower_red的值，图像值变为0
# 第三个参数：upper_red指的是图像中高于这个upper_red的值，图像值变为0
# 而在lower_red～upper_red之间的值变成255

cv2.imshow('Mask', mask)

# 背景在图中用白色表示，白色区域就是要替换的部分，但是黑色区域内有白点干扰，所以进一步优化。

#腐蚀膨胀
erode=cv2.erode(mask,None,iterations=1)
cv2.imshow('erode',erode)
dilate=cv2.dilate(erode,None,iterations=1)
cv2.imshow('dilate',dilate)

# 处理后图像单独白色点消失。
#
# 替换背景色
# 遍历全部像素点，如果该颜色为dilate里面为白色（255）则说明该点所在背景区域，于是在原图img中进行颜色替换。

#遍历替换
for i in range(rows):
    for j in range(cols):
        if dilate[i,j]==255:
            img[i,j]=(0,0,255)#此处替换颜色，为BGR通道
cv2.imshow('res',img)

k = cv2.waitKey(0)
cv2.destroyAllWindows()

def main():
    pass


if __name__ == '__main__':
    main()