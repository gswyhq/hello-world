#!/usr/bin/python3
# coding: utf-8

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
'''
对图片中的人脸检测
'''
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer.yml')

font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX

idnum = 0

names = ['小宝爸爸', '小宝', '小宝妈妈']

def change_cv2_draw(image,strs,local=(0, 0),sizes=20,colour=(255, 0, 0)):
    """
    将cv图片格式转化为PIL库的格式，用PIL的方法写入中文，然后在转化为CV的格式 ; 解决cv2.putText，中文乱码问题
    :param image: cv格式图片
    :param strs: 需要添加的中文字符串
    :param local: 添加的位置
    :param sizes: 字体大小
    :param colour: 字体颜色
    :return: cv格式图片
    """
    cv2img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pilimg = Image.fromarray(cv2img)
    draw = ImageDraw.Draw(pilimg)  # 图片上打印

    # 查看中文字体(注意冒号前有空格)：
    # gswyhq@gswyhq-PC:~$ fc-list :lang=zh

    font = ImageFont.truetype("/usr/share/fonts/deepin-font-install/SimSun/SimSun.ttf",sizes, encoding="utf-8")
    draw.text(local, strs, colour, font=font)
    image = cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)
    return image

# 人脸识别分类器
faceCascade = cv2.CascadeClassifier(r'/usr/local/lib/python3.6/dist-packages/cv2/data/haarcascade_frontalface_default.xml')

# 读入彩色图片
img = cv2.imread('/home/gswyhq/Downloads/IMG_20141005_130226.jpg', cv2.IMREAD_COLOR)

# 通过设置第二个参数使得图片窗口可调节大小，默认是不可调的（cv2.WINDOW_AUTOSIZE）
# cv2.namedWindow('image', cv2.WINDOW_NORMAL)

# 建立image窗口显示图片
# cv2.imshow('image', img)

# 无限期等待输入
# k = cv2.waitKey(0)
# 如果输入ESC退出

# 转换成灰度图像
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(gray.shape)
predict_image = np.array(gray, 'uint8')
# 人脸检测
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.2,
    minNeighbors=5,
    minSize=(420, 236)
)
result = []

cv2.imwrite('/home/gswyhq/Downloads/检测.png', gray)
print(len(faces))
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    idnum, confidence = recognizer.predict(np.asarray(predict_image[y:y+h, x:x+w]))
    print('idnum, confidence', idnum, confidence)

    if confidence < 100:
        idnum = names[idnum]
        confidence = "{0}%".format(round(100 - confidence))
    else:
        idnum = "unknown"
        confidence = "{0}%".format(round(100 - confidence))

    # cv2.putText(img, str(idnum), (x+5, y-5), font, 1, (0, 0, 255), 1)
    # 解决cv2.putText，中文乱码问题
    img = change_cv2_draw(img,str(idnum),local=(x+5, y+5),sizes=120,colour=(0, 0, 255))
    cv2.putText(img, str(confidence), (x+5, y+h-5), font, 1, (0, 0, 0), 1)


# 通过设置第二个参数使得图片窗口可调节大小，默认是不可调的（cv2.WINDOW_AUTOSIZE）
cv2.namedWindow('video', cv2.WINDOW_NORMAL)

cv2.imshow('video', img)

# 无限期等待输入
k = cv2.waitKey(0)
# 如果输入ESC退出

# press 'ESC' to quit
if k == 27:
    cv2.destroyAllWindows()

def main():
    pass


if __name__ == '__main__':
    main()