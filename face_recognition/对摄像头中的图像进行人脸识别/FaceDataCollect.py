#!/usr/bin/python3
# coding: utf-8

import cv2
import os

'''
人脸数据收集
注：1.在运行该程序前，请先创建一个Facedata文件夹并和你的程序放在一个文件夹下。

友情提示：请将程序和文件打包放在一个叫人脸识别的文件夹下。可以把分类器也放入其中。

注：2.程序运行过程中，会提示你输入id，请从0开始输入，即第一个人的脸的数据id为0，第二个人的脸的数据id为1，运行一次可收集一张人脸的数据。

注：3.程序运行时间可能会比较长，可能会有几分钟，如果嫌长，可以将     #得到1000个样本后退出摄像      这个注释前的1000，改为100。

如果实在等不及，可按esc退出，但可能会导致数据不够模型精度下降。
'''
# 调用笔记本内置摄像头，所以参数为0，如果有其他的摄像头可以调整参数为1，2

cap = cv2.VideoCapture(0)

face_detector = cv2.CascadeClassifier('/usr/local/lib/python3.6/dist-packages/cv2/data/haarcascade_frontalface_default.xml')

face_id = input('\n enter user id:')

print('\n Initializing face capture. Look at the camera and wait ...')

count = 0

while True:

    # 从摄像头读取图片

    sucess, img = cap.read()

    # 转为灰度图片

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 检测人脸

    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+w), (255, 0, 0))
        count += 1

        # 保存图像
        image_name = "Facedata/User." + str(face_id) + '.' + str(count) + '.jpg'
        cv2.imwrite(image_name, gray[y: y + h, x: x + w])
        print('保存图像在：{}'.format(os.path.abspath(image_name)))
        cv2.imshow('image', img)

    # 保持画面的持续。

    k = cv2.waitKey(1)

    if k == 27:   # 通过esc键退出摄像
        break

    elif count >= 500:  # 得到1000个样本后退出摄像
        break

# 关闭摄像头
cap.release()
cv2.destroyAllWindows()

def main():
    pass


if __name__ == '__main__':
    main()