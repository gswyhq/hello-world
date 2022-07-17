#!/usr/bin/python3
# coding: utf-8

# Application模块
# Keras中的Application模块中有一系列基于ImageNet的预训练好的图像分类模型，这些模型如下：
#
# Xception
# VGG16
# VGG19
# ResNet50
# InceptionV3
# InceptionResNetV2
# MobileNet
# DenseNet
# NASNet
# MobileNetV2
# 加载与使用这些预训练模型可以实现一些简单的分类，ImageNet支持1000个分类。
#
# 加载VGG16模型

# model= keras.applications.VGG16(weights= 'imagenet')
#
# 加载ResNet50模型
#
# model= keras.applications.ResNet50(weights= 'imagenet')
#


import keras
import tensorflow as tf
import cv2 as cv
import numpy as np


# 图像分类预测

def image_classification_demo(image_name="/home/gswyhq/data/flower_photos/daisy/450128527_fd35742d44.jpg"):
    """
    基于预训练ResNet50模型实现对图像分类预测
    :param image_name: 
    :return: 
    """
    model= keras.applications.ResNet50(weights='imagenet')

    # load the image
    src = cv.imread(image_name)
    img = cv.resize(src, (224, 224))
    img = np.expand_dims(img, 0)
    proba = model.predict(img)
    result = tf.keras.applications.resnet50.decode_predictions(proba)

    print(result)
    cv.putText(src, result[ 0][ 0][ 1],(50, 50), cv.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2, 8)
    cv.imshow("input", src)
    cv.waitKey(0)
    cv.destroyAllWindows()

# 基于VGG16迁移学习
# 
# 数据集下载
# 
# http://download.tensorflow.org/example_images/flower_photos.tgz
# 
# 5种花卉类型，接近4000张图像，分为训练集与测试集。
# 
# 通过Keras的ImageDataGenerator加载数据集，代码如下
def train(train_dir='/home/gswyhq/data/flower_photos/train_data', 
          val_dir='/home/gswyhq/data/flower_photos/validation_data',
          save_model_path="./my_transfer_vgg.h5"):
    num_classes = 5
    
    train_datagen = keras.preprocessing.image.ImageDataGenerator(
                                                                rescale=1./255,
                                                                shear_range=0.2,
                                                                zoom_range=0.2,
                                                                horizontal_flip=True)
    
    train_generator = train_datagen.flow_from_directory(train_dir,
                                                        target_size=(64, 64),
                                                        batch_size=4,
                                                        shuffle=True,
                                                        class_mode='categorical')
    
    print(train_generator.classes)
    
    print(train_generator.class_indices)
    
    test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    
    validation_generator = test_datagen.flow_from_directory(val_dir,
                                                            target_size=(64, 64),
                                                            batch_size=4,
                                                            class_mode='categorical')
    
    # 构建迁移学习网络
    # 使用VGG6的前面两个权重block，依赖block2_pool的输出，输入张量（64x64x3）
    # 构建网络的层
    # 加载VGG16模型但是不包括输出层
    input_tensor = keras.Input(shape=(64, 64, 3))
    vgg_model = keras.applications.VGG16(weights='imagenet', include_top=False, input_tensor=input_tensor)
    vgg_model.summary()
    # 显示模型加载以后的结构

    layer_dict = dict([(layer.name, layer) for layer in vgg_model.layers])
    x = layer_dict['block2_pool'].output
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(4096, activation='relu')(x)
    x = keras.layers.Dropout(0.25)(x)
    x = keras.layers.Dense(num_classes, activation=tf.nn.softmax)(x)
    custom_model = keras.models.Model(inputs=vgg_model.input, outputs=x)
    custom_model.summary()
    
    # 是否fine-tuning整个网络或者几层
    
    for layer in custom_model.layers[:7]:
        layer.trainable = True

    # 训练与保存模型
    # 编译与训练
    
    custom_model.compile(loss= 'categorical_crossentropy',
                        optimizer=tf.train.AdamOptimizer(0.0001),
                        metrics=[ 'accuracy'])
    
    custom_model.fit_generator(train_generator,
                               steps_per_epoch = 200,
                               epochs=10,
                               validation_data=validation_generator,
                               validation_steps = 20,
                               shuffle=True,
                               )
    
    # 保存整个模型
    
    custom_model.save(save_model_path)

# 使用模型测试花卉种类预测



def flowers_demo(model_path="./my_transfer_vgg.h5", test_dir='/home/gswyhq/data/flower_photos/tulips'):

    # 加载与使用
    flower_dict = [ 'daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
    new_model = keras.models.load_model( model_path)
    new_model.summary()

    for file in os.listdir(test_dir):

        src = cv.imread(os.path.join(root_dir, file))
        img = cv.resize(src, ( 64, 64))
        img = np.expand_dims(img, 0)
        result = new_model.predict(img)
        index = np.argmax(result)
        print(result, index)
        cv.putText(src, flower_dict[index],( 50, 50), cv.FONT_HERSHEY_PLAIN, 2.0, ( 0, 0, 255), 2, 8)
        cv.imshow( "input", src)
        cv.waitKey( 0)
        cv.destroyAllWindows()

def main():
    train(train_dir='/home/gswyhq/data/flower_photos/train_data',
          val_dir='/home/gswyhq/data/flower_photos/validation_data',
          save_model_path="./my_transfer_vgg.h5")


if __name__ == '__main__':
    main()