
# 对笔记本摄像头捕获到的人脸进行识别

## 1.FaceDetection,人脸检测
一个蓝色的矩形框住你的脸，两个绿色的矩形框住你的眼睛，按esc可退出。

## 2.FaceDataCollect,人脸数据收集
运行该程序前，请先创建一个Facedata文件夹并和你的程序放在一个文件夹下。
程序运行过程中，会提示你输入id，请从0开始输入，即第一个人的脸的数据id为0，第二个人的脸的数据id为1，运行一次可收集一张人脸的数据。
程序运行时间可能会比较长，可能会有几分钟，如果嫌长，可以将     “#得到1000个样本后退出摄像”      这个注释前的1000，改为100。
如果实在等不及，可按esc退出，但可能会导致数据不够模型精度下降。

## 3.face_training,人脸数据训练
LBPHFaceRecognizer_create()为contrib中的函数。
读取Facedata目录下的数据进行训练。
训练结果保存在trainer.yml文件中。

## 4.face_recognition 人脸识别
names中存储人的名字，若该人id为0则他的名字在第一位，id位1则排在第二位，以此类推。
最终效果为一个绿框，框住人脸，左上角为红色的人名，左下角为黑色的概率。

## 其他
有时候，预测出来的结果都是一样的；除了程序上的问题外，可能是因为训练数据的问题；
更换下训练数据即可，如采用数据：https://github.com/informramiz/opencv-face-recognition-python/tree/master/training-data

[参考资料](https://www.cnblogs.com/xp12345/p/9818435.html)