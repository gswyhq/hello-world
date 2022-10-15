#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import tensorflow as tf
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, \
    AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from keras.utils.vis_utils import model_to_dot
# from keras.utils import plot_model
from keras.utils.vis_utils import plot_model
from keras.initializers import glorot_uniform
from tensorflow.python.keras.utils import np_utils
import pydot
from IPython.display import SVG
import scipy.misc
from matplotlib.pyplot import imshow
import keras.backend as K
from IPython.display import Image

K.set_image_data_format('channels_last')
K.set_learning_phase(1)


def identify_block(X, f, filters, stage, block):
    """
    恒等块
    基本结构：可以看到在输入X卷积二次之后输出的结果和输入X进行了相加，然后进行了激活。这样做就实现了更深层次的梯度直接传向较浅的层的功能。
    实现细节：由于需要相加，那么两次卷积的输出结果需要和输入X的shape相同，所以这就被称为恒等块。下面的实现中将会完成下图的3层跳跃，同样这也是一个恒等块。
    X - 输入的tensor类型数据，维度为（m, n_H_prev, n_W_prev, n_H_prev）
    f - kernal大小
    filters - 整数列表，定义每一层卷积层过滤器的数量
    stage - 整数 定义层位置
    block - 字符串 定义层位置

    X - 恒等输出，tensor类型，维度（n_H, n_W, n_C）
    """
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    F1, F2, F3 = filters  # 定义输出特征的个数
    X_shortcut = X

    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2a',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)
    # 没有激活

    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    return X


def convolutional_block(X, f, filters, stage, block, s=2):
    '''卷积块，上述恒等块要求在主线上进行卷积时shape不变，这样才能和捷径上的X相加。如果形状变化了，那就在捷径中加上卷积层，使捷径上卷积层的输出和主线上的shape相同。'''
    # 参数意义和上文相同
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    F1, F2, F3 = filters
    X_shortcut = X

    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '2a',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    # shortcut
    X_shortcut = Conv2D(filters=F3, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '1',
                        kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)

    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X


def ResNet50(input_shape=(64, 64, 3), classes=6):
    """
    ID_BLOCK对应恒等块，CONV_BLOCK对应卷积块，每个块有3层，总共50层。
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

    input_shape: 据集维度
    classes： 分类数
    """
    # 定义一个placeholder
    X_input = Input(input_shape)
    # 0填充
    X = ZeroPadding2D((3, 3))(X_input)

    # stage1
    X = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), name='conv1', kernel_initializer=glorot_uniform(seed=0))(
        X)
    X = BatchNormalization(axis=3, name='bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(X)

    # stage2
    X = convolutional_block(X, f=3, filters=[64, 64, 256], stage=2, block='a', s=1)
    X = identify_block(X, f=3, filters=[64, 64, 256], stage=2, block='b')
    X = identify_block(X, f=3, filters=[64, 64, 256], stage=2, block='c')

    # stage3
    X = convolutional_block(X, f=3, filters=[128, 128, 512], stage=3, block="a", s=2)
    X = identify_block(X, f=3, filters=[128, 128, 512], stage=3, block="b")
    X = identify_block(X, f=3, filters=[128, 128, 512], stage=3, block="c")
    X = identify_block(X, f=3, filters=[128, 128, 512], stage=3, block="d")

    # stage4
    X = convolutional_block(X, f=3, filters=[256, 256, 1024], stage=4, block="a", s=2)
    X = identify_block(X, f=3, filters=[256, 256, 1024], stage=4, block="b")
    X = identify_block(X, f=3, filters=[256, 256, 1024], stage=4, block="c")
    X = identify_block(X, f=3, filters=[256, 256, 1024], stage=4, block="d")
    X = identify_block(X, f=3, filters=[256, 256, 1024], stage=4, block="e")
    X = identify_block(X, f=3, filters=[256, 256, 1024], stage=4, block="f")

    # stage5
    X = convolutional_block(X, f=3, filters=[512, 512, 2048], stage=5, block="a", s=2)
    X = identify_block(X, f=3, filters=[512, 512, 2048], stage=5, block="b")
    X = identify_block(X, f=3, filters=[512, 512, 2048], stage=5, block="c")

    # 均值池化
    X = AveragePooling2D(pool_size=(2, 2), padding='same')(X)

    # 输出层
    X = Flatten()(X)
    X = Dense(classes, activation="softmax", name='fc' + str(classes), kernel_initializer=glorot_uniform(seed=0))(X)

    model = Model(inputs=X_input, outputs=X, name='ResNet50')
    return model


# 创建实例以及编译 ，训练。我们要做的就是输入数据的shape
X_train = np.random.randint(255, size=(100, 64, 64, 3))
Y_train = np_utils.to_categorical([np.random.randint(0, 6) for _ in range(100)], 6)
X_test = np.random.randint(255, size=(20, 64, 64, 3))
Y_test = np_utils.to_categorical([np.random.randint(0, 6) for _ in range(20)], 6)
model = ResNet50(input_shape=(64, 64, 3), classes=6)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

# 训练模型
model.fit(X_train, Y_train, epochs=2, batch_size=32)

# 模型评估
preds = model.evaluate(X_test, Y_test)

print("误差值 = " + str(preds[0]))
print("准确率 = " + str(preds[1]))


# 展示模型结构
plot_model(model, to_file='model.png')
SVG(model_to_dot(model).create(prog='dot', format='svg'))


# 单例测试
img_path = 'result/images/20220705142707.jpg'

my_image = image.load_img(img_path, target_size=(64, 64))
my_image = image.img_to_array(my_image)

my_image = np.expand_dims(my_image, axis=0)
my_image = preprocess_input(my_image)

print("my_image.shape = " + str(my_image.shape))

print("class prediction vector [p(0), p(1), p(2), p(3), p(4), p(5)] = ")
print(model.predict(my_image))

# 展示图片
Image(img_path)

def main():
    pass


if __name__ == '__main__':
    main()
