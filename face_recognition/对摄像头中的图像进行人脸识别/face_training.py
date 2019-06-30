#!/usr/bin/python3
# coding: utf-8

import numpy as np
from PIL import Image
import os
import cv2
# 人脸数据路径
path = 'Facedata'

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("/usr/local/lib/python3.6/dist-packages/cv2/data/haarcascade_frontalface_default.xml")

def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faceSamples = []
    ids = []
    # count = 0
    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L')   # convert it to grayscale
        img_numpy = np.array(PIL_img, 'uint8')

        # 上方法读取图像，等同于：
        # im = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
        # img_numpy = np.asarray(im, dtype=np.uint8)

        id = int(os.path.split(imagePath)[-1].split(".")[1])
        print(img_numpy)

        w, h = img_numpy.shape
        faces = detector.detectMultiScale(img_numpy, scaleFactor=1.05, minNeighbors=3, minSize=(w//2, h//2))
        # if len(faces) != 1:
        #     print('发现人脸数', len(faces), imagePath)
        #     count += 1
            # # 画矩形
            # img = img_numpy
            # for (x, y, w, h) in faces:
            #     cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # cv2.imshow('video', img)
            #
            # k = cv2.waitKey(0)
            # if k == 27:  # press 'ESC' to quit
            #     break
        for (x, y, w, h) in faces:
            faceSamples.append(img_numpy[y:y + h, x: x + w])
            ids.append(id)
    # print(count)
    return faceSamples, ids


# cv2.destroyAllWindows()
print('Training faces. It will take a few seconds. Wait ...')
faces, ids = getImagesAndLabels(path)
recognizer.train(faces, np.asarray(ids))

recognizer.write(r'trainer.yml')
print("{0} faces trained. Exiting Program".format(len(np.unique(ids))))

def main():
    pass


if __name__ == '__main__':
    main()