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
    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L')   # convert it to grayscale
        img_numpy = np.array(PIL_img, 'uint8')
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)
        for (x, y, w, h) in faces:
            faceSamples.append(img_numpy[y:y + h, x: x + w])
            ids.append(id)
    return faceSamples, ids


print('Training faces. It will take a few seconds. Wait ...')
faces, ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids))

recognizer.write(r'trainer.yml')
print("{0} faces trained. Exiting Program".format(len(np.unique(ids))))

def main():
    pass


if __name__ == '__main__':
    main()