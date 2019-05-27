#!/usr/bin/python3
# coding: utf-8

# https://mp.weixin.qq.com/s/AZpRv0zrgQPVE3El-9Vs2g

# 图像导入&显示
import cv2

image = cv2.imread('/home/gswyhq/Downloads/星空.jpeg')
cv2.imshow('Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 警告1: 通过openCV读取图像时,它不是以RGB 颜色空间来读取,而是以BGR 颜色空间。有时候这
# 对你来说不是问题,只有当你想在图片中添加一些颜色时,你才会遇到问题。
# 有两种解决方案:
# 1. 将R — 第一个颜色值(红色)和B — 第三个颜色值(蓝色) 交换, 这样红色就是 (0,0,255) 而不是
# (255,0,0)。
# 2. 将颜色空间变成RGB:
# rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# 使用rgb_image代替image继续处理代码。
# 警告2: 要关闭显示图像的窗口,请按任意按钮。如果你使用关闭按钮,它可能会导致窗口冻结(我在
# Jupyter笔记本上运行代码时发生了这种情况)。
# 为了简单起见,在整个教程中,我将使用这种方法来查看图像:

def viewImage(image, name_of_window):
    cv2.namedWindow(name_of_window, cv2.WINDOW_NORMAL)
    cv2.imshow(name_of_window, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 对图像进行裁剪：
cropped = image[10:500, 500:2000]
viewImage(cropped, "裁剪之后的图像")
# 其中: image[10:500,500:200] 是 image[y:y+h,x:x+w]。
# 调整大小

# 调整大小到20%后
scale_percent = 20 # 原始大小的百分比
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[1] * scale_percent / 100)
dim = (width, height)

resized = cv2.resize(image, dim , interpolation=cv2.INTER_AREA)
viewImage(resized, "裁剪20%后")

# 进行180度旋转
(h, w, d) = image.shape
center = (w//2, h//2)
M = cv2.getRotationMatrix2D(center, 180, 1.0)
rotated = cv2.warpAffine(image, M, (w, h))
viewImage(rotated, "旋转180度后")

# image.shape输出高度、宽度和通道。M是旋转矩阵——它将图像围绕其中心旋转180度。
# -ve表示顺时针旋转图像的角度 & +ve逆表示逆时针旋转图像的角度。


# 灰度和阈值(黑白效果)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret, threshold_image = cv2.threshold(image, 127, 255, 0)
viewImage(gray_image, '灰度与阈值')
viewImage(threshold_image, '黑白效果')

# gray_image 是灰度图像的单通道版本。
# 这个threshold函数将把所有比127深(小)的像素点阴影值设定为0,所有比127亮(大)的像素点阴影
# 值设定为255。

# 另一个例子；
ret, threshold = cv2.threshold(image, 150, 200, 10)
# 这将把所有阴影值小于150的像素点设定为10和所有大于150的像素点设定为200。


# 模糊/平滑
blurred = cv2.GaussianBlur(image, (51, 51), 0)
viewImage(blurred, '模糊')
# 高斯模糊函数接受3个参数:
# 1. 第一个参数是要模糊的图像。
# 2. 第二个参数必须是一个由两个正奇数组成的元组。当它们增加,模糊效果也会增加。
# 3. 第三个参数是sigmaX和sigmaY。当左边位于0时,它们会自动从内部大小计算出来。

# 更多关于模糊函数的内容,请查看这里。
# (https://docs.opencv.org/3.1.0/d4/d13/tutorial_py_filtering.html )


# 在图像上绘制矩形框或边框
output = image.copy()
cv2.rectangle(output, (2600, 800), (4100, 2400), (0, 255, 255), 10)
viewImage(output, '在图像上绘制矩形框或边框')

# rectangle函数接受5个参数:
# 1. 第一个参数是图像。
# 2. 第二个参数是x1, y1 -左上角坐标。
# 3. 第三个参数是x2, y2 -右下角坐标。
# 4. 第四个参数是矩形颜色(GBR/RGB,取决于你如何导入图像)。
# 5. 第五个参数是矩形线宽。


# 绘制一条线
output = image.copy()
cv2.line(output, (60, 20), (400, 200), (0, 0, 255), 5)
viewImage(output, '绘制一条线')

# line函数接受5个参数:
# 第一个参数是要画的线所在的图像。
# 第二个参数是x1, y1。
# 第三个参数是x2, y2。
# 第四个参数是线条颜色(GBR/RGB,取决于你如何导入图像)。
# 第五个参数是线宽。

# 在图片上写入文字
output = image.copy()
cv2.putText(output, "文本", (1500, 3600), cv2.FONT_HERSHEY_SIMPLEX, 15, (30, 105, 210), 40)
viewImage(output, '在图片上写入文字')
# putText函数接受 七个参数：
# 第一个参数是要写入文本的图像。
# 第二个参数是待写入文本。
# 第三个参数是x, y——文本开始的左下角坐标。
# 第四个参数是字体类型。
# 第五个参数是字体大小。
# 第六个参数是颜色(GBR/RGB，取决于你如何导入图像)。
# 第七个参数是文本线条的粗细。


# 人脸检测
image_path = '/home/gswyhq/Downloads/温馨小贴士.jpeg'
face_casade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = face_casade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(10, 10)
)

face_detected = format(len(faces)) + 'faces detected!'
print(face_detected)

for (x,y, w,h ) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 255, 0), 2)
viewImage(image, face_detected)

# detectMultiScale函数是一种检测对象的通用函数。因为我们调用的是人脸级联，所以它会检测到人脸。
# detectMultiScale函数接受4个参数：
# 第一个参数是灰阶图像。
# 第二个参数是scaleFactor。因为有些人脸可能离镜头更近，所以看起来会比后台的人脸更大。比例系数弥补了这一点。
# 检测算法使用一个移动窗口来检测对象。minNeighbors定义在当前对象附近检测到多少对象，然后再声明检测到人脸。
# 与此同时，minsize给出了每个窗口的大小。
# 轮廓——一种对象检测方法
# 使用基于颜色的图像分割，你可以来检测对象。
# cv2.findContours & cv2.drawContours 这两个函数可以帮助你做到这一点。
# 最近，我写了一篇非常详细的文章，叫做《使用Python通过基于颜色的图像分割来进行对象检测》。你需要知道的关于轮廓的一切都在那里。
# （https://towardsdatascience.com/object-detection-via-color-based-image-segmentation-using-python-e9b7c72f0e11  ）
# 最终,保存图片

cv2.imwrite('./保存图像.jpg', image)

def main():
    pass


if __name__ == '__main__':
    main()