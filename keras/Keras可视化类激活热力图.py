#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 参考资料：https://blog.csdn.net/Forrest97/article/details/105895591
# https://blog.csdn.net/weixin_40356612/article/details/107866835

##########类激活图（CAM，class activation map）可视化。
import tensorflow.keras.backend as K
import tensorflow as tf
from keras.applications.vgg16 import VGG16
from keras.applications.resnet import ResNet50
model=VGG16(weights='imagenet')#这次网络中没有舍弃密集连结分类
# 也可以事先下载好预训练模型
# https://storage.googleapis.com/tensorflow/keras-applications/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels.h5 > ~/.keras/models/vgg16_weights_tf_dim_ordering_tf_kernels.h5
# https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json ->
# model.summary()
# Model: "vgg16"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# input_2 (InputLayer)         [(None, 224, 224, 3)]     0
# _________________________________________________________________
# block1_conv1 (Conv2D)        (None, 224, 224, 64)      1792
# _________________________________________________________________
# block1_conv2 (Conv2D)        (None, 224, 224, 64)      36928
# _________________________________________________________________
# block1_pool (MaxPooling2D)   (None, 112, 112, 64)      0
# _________________________________________________________________
# block2_conv1 (Conv2D)        (None, 112, 112, 128)     73856
# _________________________________________________________________
# block2_conv2 (Conv2D)        (None, 112, 112, 128)     147584
# _________________________________________________________________
# block2_pool (MaxPooling2D)   (None, 56, 56, 128)       0
# _________________________________________________________________
# block3_conv1 (Conv2D)        (None, 56, 56, 256)       295168
# _________________________________________________________________
# block3_conv2 (Conv2D)        (None, 56, 56, 256)       590080
# _________________________________________________________________
# block3_conv3 (Conv2D)        (None, 56, 56, 256)       590080
# _________________________________________________________________
# block3_pool (MaxPooling2D)   (None, 28, 28, 256)       0
# _________________________________________________________________
# block4_conv1 (Conv2D)        (None, 28, 28, 512)       1180160
# _________________________________________________________________
# block4_conv2 (Conv2D)        (None, 28, 28, 512)       2359808
# _________________________________________________________________
# block4_conv3 (Conv2D)        (None, 28, 28, 512)       2359808
# _________________________________________________________________
# block4_pool (MaxPooling2D)   (None, 14, 14, 512)       0
# _________________________________________________________________
# block5_conv1 (Conv2D)        (None, 14, 14, 512)       2359808
# _________________________________________________________________
# block5_conv2 (Conv2D)        (None, 14, 14, 512)       2359808
# _________________________________________________________________
# block5_conv3 (Conv2D)        (None, 14, 14, 512)       2359808
# _________________________________________________________________
# block5_pool (MaxPooling2D)   (None, 7, 7, 512)         0
# _________________________________________________________________
# flatten (Flatten)            (None, 25088)             0
# _________________________________________________________________
# fc1 (Dense)                  (None, 4096)              102764544
# _________________________________________________________________
# fc2 (Dense)                  (None, 4096)              16781312
# _________________________________________________________________
# predictions (Dense)          (None, 1000)              4097000
# =================================================================
# Total params: 138,357,544
# Trainable params: 138,357,544
# Non-trainable params: 0
# _________________________________________________________________

#加载图像使得大小调整为224*224#带分类器识别出是哪个物体
from keras.preprocessing import image
from keras import backend as K
from keras.applications.vgg16 import preprocess_input,decode_predictions
import numpy as np
# img_path = './data/creative_commons_elephant.jpg'
img_path = r'D:\Users\Downloads\elephant.jpeg'
img=image.load_img(img_path,target_size=(224,224))
x=image.img_to_array(img)
x=np.expand_dims(x,axis=0)#增加一个维度（1，224，224，3）
#卷积神经网络都是4维的
x=preprocess_input(x)#对批量进行预处理，按通道进行颜色标准化
preds=model.predict(x)
print('预测结果：',decode_predictions(preds,top=3)[0])  # decode_predictions函数，解码模型的预测结果，返回指定top个类的预测元组(class_name, class_description, score)。
# 预测结果： [('n02504458', 'African_elephant', 0.9166607), ('n01871265', 'tusker', 0.0786288), ('n02504013', 'Indian_elephant', 0.0045627384)]
print('索引编号：',np.argmax(preds[0]))#386
# 预测向量中被最大激活的元素是对应非洲象类别元素，其索引编号为：386

#######为了展示图像中那些部分是非洲象我们使用Grad_Cam算法#######

last_conv_layer=model.get_layer('block5_conv3')
heatmap_model =models.Model([model.inputs], [last_conv_layer.output, model.output])

# Grad-CAM 算法实现
with tf.GradientTape() as gtape:
    conv_output, Predictions = heatmap_model(x)
    prob = Predictions[:, np.argmax(Predictions[0])] # 最大可能性类别的预测概率
    grads = gtape.gradient(prob, conv_output)  # 类别与卷积层的梯度 (1,14,14,512)
    pooled_grads = K.mean(grads, axis=(0,1,2)) # 特征层梯度的全局平均代表每个特征层权重
heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_output), axis=-1) #权重与特征层相乘，512层求和平均

# 绘制激活热力图
######为了可视化，将热力图标准化到0-1范围内##############################
heatmap = np.maximum(heatmap, 0)
max_heat = np.max(heatmap)
if max_heat == 0:
    max_heat = 1e-10
heatmap /= max_heat
plt.matshow(heatmap[0], cmap='viridis')

# 与原有图片进行加权叠加
###########用openCv来生成一张图片，然后将原始图像叠加在刚刚生成的热力图上########
import cv2
original_img=cv2.imread(img_path)#加载原始图像

# 将热力图调整为与原始图像一样大小
heatmap1 = cv2.resize(heatmap[0], (original_img.shape[1], original_img.shape[0]), interpolation=cv2.INTER_CUBIC)
heatmap1 = np.uint8(255*heatmap1) # 将热力图改为rgb格式
heatmap1 = cv2.applyColorMap(heatmap1, cv2.COLORMAP_JET) # #将热力图应用于原始图像
frame_out=cv2.addWeighted(original_img,0.5,heatmap1,0.5,0) #0.5是热力图的强度因子
cv2.imwrite('大象_cam.jpg', frame_out) # #不支持中文,可以写入但是文件名乱码

plt.figure()
plt.imshow(frame_out)

def main():
    pass


if __name__ == '__main__':
    main()