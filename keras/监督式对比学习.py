#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 介绍
# 监督对比学习（Supervised Contrastive Learning）是一种在分类任务上优于交叉熵监督训练的训练方法。
# 本质上，用监督对比学习训练图像分类模型分两个阶段进行：
# 训练编码器以学习产生输入图像的向量表示，使得同一类中的图像的表示与不同类中图像的表示相比将更相似。
#
# 在冻结的编码器上训练分类器。

# https://keras.io/examples/vision/supervised-contrastive-learning/

import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

num_classes = 10
input_shape = (32, 32, 3)

# 加载训练测试数据集
# (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
(x_train, y_train), (x_test, y_test) = (np.random.random(size=(2000, 32, 32, 3)), np.random.randint(10, size=(2000, 1))), (np.random.random(size=(500, 32, 32, 3)), np.random.randint(10, size=(500, 1)))

# Display shapes of train and test datasets
print(f"x_train shape: {x_train.shape} - y_train shape: {y_train.shape}")
print(f"x_test shape: {x_test.shape} - y_test shape: {y_test.shape}")
# x_train shape: (50000, 32, 32, 3) - y_train shape: (50000, 1)
# x_test shape: (10000, 32, 32, 3) - y_test shape: (10000, 1)

# 数据增强
data_augmentation = keras.Sequential(
    [
        layers.Normalization(),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.02),
        layers.RandomWidth(0.2),
        layers.RandomHeight(0.2),
    ]
)

# 设置normalization layer状态
data_augmentation.layers[0].adapt(x_train)
# 构建编码器模型
# 编码器模型将图像作为输入，并将其转换为2048维特征向量，2048-dimensional feature vector.

def create_encoder():
    resnet = keras.applications.ResNet50V2(
        include_top=False, weights=None, input_shape=input_shape, pooling="avg"
    )

    inputs = keras.Input(shape=input_shape)
    augmented = data_augmentation(inputs)
    outputs = resnet(augmented)
    model = keras.Model(inputs=inputs, outputs=outputs, name="cifar10-encoder")
    return model


encoder = create_encoder()
encoder.summary()

learning_rate = 0.001
batch_size = 265
hidden_units = 512
projection_units = 128
num_epochs = 50
dropout_rate = 0.5
temperature = 0.05
# Model: "cifar10-encoder"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# input_2 (InputLayer)         [(None, 32, 32, 3)]       0
# _________________________________________________________________
# sequential (Sequential)      (None, None, None, 3)     7
# _________________________________________________________________
# resnet50v2 (Functional)      (None, 2048)              23564800
# =================================================================
# Total params: 23,564,807
# Trainable params: 23,519,360
# Non-trainable params: 45,447
# _________________________________________________________________


# 构建分类模型
# 分类模型在编码器顶部添加一个全连接层，并添加一个softmax层；

def create_classifier(encoder, trainable=True):

    for layer in encoder.layers:
        layer.trainable = trainable

    inputs = keras.Input(shape=input_shape)
    features = encoder(inputs)
    features = layers.Dropout(dropout_rate)(features)
    features = layers.Dense(hidden_units, activation="relu")(features)
    features = layers.Dropout(dropout_rate)(features)
    outputs = layers.Dense(num_classes, activation="softmax")(features)

    model = keras.Model(inputs=inputs, outputs=outputs, name="cifar10-classifier")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )
    return model

# 实验1：训练基线分类模型
# 在本实验中，基线分类器照常训练，即编码器和分类器部分作为单个模型一起训练，以最小化交叉熵损失。

encoder = create_encoder()
classifier = create_classifier(encoder)
classifier.summary()

history = classifier.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=num_epochs)

accuracy = classifier.evaluate(x_test, y_test)[1]
print(f"Test accuracy: {round(accuracy * 100, 2)}%")
# Model: "cifar10-classifier"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# input_5 (InputLayer)         [(None, 32, 32, 3)]       0
# _________________________________________________________________
# cifar10-encoder (Functional) (None, 2048)              23564807
# _________________________________________________________________
# dropout (Dropout)            (None, 2048)              0
# _________________________________________________________________
# dense (Dense)                (None, 512)               1049088
# _________________________________________________________________
# dropout_1 (Dropout)          (None, 512)               0
# _________________________________________________________________
# dense_1 (Dense)              (None, 10)                5130
# =================================================================
# Total params: 24,619,025
# Trainable params: 24,573,578
# Non-trainable params: 45,447
# _________________________________________________________________
# Epoch 1/50
# 189/189 [==============================] - 15s 77ms/step - loss: 1.9369 - sparse_categorical_accuracy: 0.2874
# Epoch 2/50
# 189/189 [==============================] - 11s 57ms/step - loss: 1.5133 - sparse_categorical_accuracy: 0.4505
# Epoch 3/50
# 189/189 [==============================] - 11s 57ms/step - loss: 1.3468 - sparse_categorical_accuracy: 0.5204
# Epoch 4/50
# 189/189 [==============================] - 11s 60ms/step - loss: 1.2159 - sparse_categorical_accuracy: 0.5733
# Epoch 5/50
# 189/189 [==============================] - 11s 56ms/step - loss: 1.1516 - sparse_categorical_accuracy: 0.6032
# Epoch 6/50
# 189/189 [==============================] - 11s 58ms/step - loss: 1.0769 - sparse_categorical_accuracy: 0.6254
# Epoch 7/50
# 189/189 [==============================] - 11s 58ms/step - loss: 0.9964 - sparse_categorical_accuracy: 0.6547
# Epoch 8/50
# 189/189 [==============================] - 10s 55ms/step - loss: 0.9563 - sparse_categorical_accuracy: 0.6703
# Epoch 9/50
# 189/189 [==============================] - 10s 55ms/step - loss: 0.8952 - sparse_categorical_accuracy: 0.6925
# Epoch 10/50
# 189/189 [==============================] - 11s 56ms/step - loss: 0.8986 - sparse_categorical_accuracy: 0.6922
# Epoch 11/50
# 189/189 [==============================] - 10s 55ms/step - loss: 0.8381 - sparse_categorical_accuracy: 0.7145
# Epoch 12/50
# 189/189 [==============================] - 10s 55ms/step - loss: 0.8513 - sparse_categorical_accuracy: 0.7086
# Epoch 13/50
# 189/189 [==============================] - 11s 56ms/step - loss: 0.7557 - sparse_categorical_accuracy: 0.7448
# Epoch 14/50
# 189/189 [==============================] - 11s 56ms/step - loss: 0.7168 - sparse_categorical_accuracy: 0.7548
# Epoch 15/50
# 189/189 [==============================] - 10s 55ms/step - loss: 0.6772 - sparse_categorical_accuracy: 0.7690
# Epoch 16/50
# 189/189 [==============================] - 11s 56ms/step - loss: 0.7587 - sparse_categorical_accuracy: 0.7416
# Epoch 17/50
# 189/189 [==============================] - 10s 55ms/step - loss: 0.6873 - sparse_categorical_accuracy: 0.7665
# Epoch 18/50
# 189/189 [==============================] - 11s 56ms/step - loss: 0.6418 - sparse_categorical_accuracy: 0.7804
# Epoch 19/50
# 189/189 [==============================] - 11s 56ms/step - loss: 0.6086 - sparse_categorical_accuracy: 0.7927
# Epoch 20/50
# 189/189 [==============================] - 10s 55ms/step - loss: 0.5903 - sparse_categorical_accuracy: 0.7978
# Epoch 21/50
# 189/189 [==============================] - 11s 56ms/step - loss: 0.5636 - sparse_categorical_accuracy: 0.8083
# Epoch 22/50
# 189/189 [==============================] - 11s 56ms/step - loss: 0.5527 - sparse_categorical_accuracy: 0.8123
# Epoch 23/50
# 189/189 [==============================] - 11s 56ms/step - loss: 0.5308 - sparse_categorical_accuracy: 0.8191
# Epoch 24/50
# 189/189 [==============================] - 10s 55ms/step - loss: 0.5282 - sparse_categorical_accuracy: 0.8223
# Epoch 25/50
# 189/189 [==============================] - 10s 55ms/step - loss: 0.5090 - sparse_categorical_accuracy: 0.8263
# Epoch 26/50
# 189/189 [==============================] - 10s 55ms/step - loss: 0.5497 - sparse_categorical_accuracy: 0.8181
# Epoch 27/50
# 189/189 [==============================] - 10s 55ms/step - loss: 0.4950 - sparse_categorical_accuracy: 0.8332
# Epoch 28/50
# 189/189 [==============================] - 11s 56ms/step - loss: 0.4727 - sparse_categorical_accuracy: 0.8391
# Epoch 29/50
# 167/189 [=========================>....] - ETA: 1s - loss: 0.4594 - sparse_categorical_accuracy: 0.8444


# 实验 2: 使用监督对比学习 supervised contrastive learning
# 本实验分两个阶段，第一阶段，对编码器进行预训练以优化监督对比损失；
# 在第二阶段，使用训练的编码器对分类器进行训练，并冻结编码器权重，只有具有softmax的完全连接层的权重被优化

# 1. 监督对比学习的损失函数
class SupervisedContrastiveLoss(keras.losses.Loss):
    def __init__(self, temperature=1, name=None):
        super(SupervisedContrastiveLoss, self).__init__(name=name)
        self.temperature = temperature

    def __call__(self, labels, feature_vectors, sample_weight=None):
        # Normalize feature vectors
        feature_vectors_normalized = tf.math.l2_normalize(feature_vectors, axis=1)
        # Compute logits
        logits = tf.divide(
            tf.matmul(
                feature_vectors_normalized, tf.transpose(feature_vectors_normalized)
            ),
            self.temperature,
        )
        return tfa.losses.npairs_loss(tf.squeeze(labels), logits)


def add_projection_head(encoder):
    inputs = keras.Input(shape=input_shape)
    features = encoder(inputs)
    outputs = layers.Dense(projection_units, activation="relu")(features)
    model = keras.Model(
        inputs=inputs, outputs=outputs, name="cifar-encoder_with_projection-head"
    )
    return model

# 2. 预训练编码器
encoder = create_encoder()

encoder_with_projection_head = add_projection_head(encoder)
encoder_with_projection_head.compile(
    optimizer=keras.optimizers.Adam(learning_rate),
    loss=SupervisedContrastiveLoss(temperature),
)

encoder_with_projection_head.summary()

history = encoder_with_projection_head.fit(
    x=x_train, y=y_train, batch_size=batch_size, epochs=num_epochs
)
# Model: "cifar-encoder_with_projection-head"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# input_8 (InputLayer)         [(None, 32, 32, 3)]       0
# _________________________________________________________________
# cifar10-encoder (Functional) (None, 2048)              23564807
# _________________________________________________________________
# dense_2 (Dense)              (None, 128)               262272
# =================================================================
# Total params: 23,827,079
# Trainable params: 23,781,632
# Non-trainable params: 45,447
# _________________________________________________________________
# Epoch 1/50
# 189/189 [==============================] - 11s 56ms/step - loss: 5.3730
# Epoch 2/50
# 189/189 [==============================] - 11s 56ms/step - loss: 5.1583
# Epoch 3/50
# 189/189 [==============================] - 10s 55ms/step - loss: 5.0368
# Epoch 4/50
# 189/189 [==============================] - 11s 56ms/step - loss: 4.9349
# Epoch 5/50
# 189/189 [==============================] - 10s 55ms/step - loss: 4.8262
# Epoch 6/50
# 189/189 [==============================] - 11s 56ms/step - loss: 4.7470
# Epoch 7/50
# 189/189 [==============================] - 11s 56ms/step - loss: 4.6835
# Epoch 8/50
# 189/189 [==============================] - 11s 56ms/step - loss: 4.6120
# Epoch 9/50
# 189/189 [==============================] - 11s 56ms/step - loss: 4.5608
# Epoch 10/50
# 189/189 [==============================] - 10s 55ms/step - loss: 4.5075
# Epoch 11/50
# 189/189 [==============================] - 11s 56ms/step - loss: 4.4674
# Epoch 12/50
# 189/189 [==============================] - 10s 56ms/step - loss: 4.4362
# Epoch 13/50
# 189/189 [==============================] - 11s 56ms/step - loss: 4.3899
# Epoch 14/50
# 189/189 [==============================] - 10s 55ms/step - loss: 4.3664
# Epoch 15/50
# 189/189 [==============================] - 11s 56ms/step - loss: 4.3188
# Epoch 16/50
# 189/189 [==============================] - 10s 56ms/step - loss: 4.3030
# Epoch 17/50
# 189/189 [==============================] - 11s 57ms/step - loss: 4.2725
# Epoch 18/50
# 189/189 [==============================] - 10s 55ms/step - loss: 4.2523
# Epoch 19/50
# 189/189 [==============================] - 11s 56ms/step - loss: 4.2100
# Epoch 20/50
# 189/189 [==============================] - 10s 55ms/step - loss: 4.2033
# Epoch 21/50
# 189/189 [==============================] - 11s 56ms/step - loss: 4.1741
# Epoch 22/50
# 189/189 [==============================] - 11s 56ms/step - loss: 4.1443
# Epoch 23/50
# 189/189 [==============================] - 11s 56ms/step - loss: 4.1350
# Epoch 24/50
# 189/189 [==============================] - 11s 57ms/step - loss: 4.1192
# Epoch 25/50
# 189/189 [==============================] - 11s 56ms/step - loss: 4.1002
# Epoch 26/50
# 189/189 [==============================] - 11s 57ms/step - loss: 4.0797
# Epoch 27/50
# 189/189 [==============================] - 11s 56ms/step - loss: 4.0547
# Epoch 28/50
# 189/189 [==============================] - 11s 56ms/step - loss: 4.0336
# Epoch 29/50
# 189/189 [==============================] - 11s 56ms/step - loss: 4.0299
# Epoch 30/50
# 189/189 [==============================] - 11s 56ms/step - loss: 4.0031
# Epoch 31/50
# 189/189 [==============================] - 11s 56ms/step - loss: 3.9979
# Epoch 32/50
# 189/189 [==============================] - 11s 56ms/step - loss: 3.9777
# Epoch 33/50
# 189/189 [==============================] - 10s 55ms/step - loss: 3.9800
# Epoch 34/50
# 189/189 [==============================] - 11s 56ms/step - loss: 3.9538
# Epoch 35/50
# 189/189 [==============================] - 11s 56ms/step - loss: 3.9298
# Epoch 36/50
# 189/189 [==============================] - 11s 57ms/step - loss: 3.9241
# Epoch 37/50
# 189/189 [==============================] - 11s 56ms/step - loss: 3.9102
# Epoch 38/50
# 189/189 [==============================] - 11s 56ms/step - loss: 3.9075
# Epoch 39/50
# 189/189 [==============================] - 11s 56ms/step - loss: 3.8897
# Epoch 40/50
# 189/189 [==============================] - 11s 57ms/step - loss: 3.8871
# Epoch 41/50
# 189/189 [==============================] - 11s 56ms/step - loss: 3.8596
# Epoch 42/50
# 189/189 [==============================] - 10s 56ms/step - loss: 3.8526
# Epoch 43/50
# 189/189 [==============================] - 11s 56ms/step - loss: 3.8417
# Epoch 44/50
# 189/189 [==============================] - 10s 55ms/step - loss: 3.8239
# Epoch 45/50
# 189/189 [==============================] - 11s 56ms/step - loss: 3.8178
# Epoch 46/50
# 189/189 [==============================] - 11s 56ms/step - loss: 3.8065
# Epoch 47/50
# 189/189 [==============================] - 11s 56ms/step - loss: 3.8185
# Epoch 48/50
# 189/189 [==============================] - 11s 56ms/step - loss: 3.8022
# Epoch 49/50
# 189/189 [==============================] - 11s 56ms/step - loss: 3.7815
# Epoch 50/50
# 189/189 [==============================] - 11s 56ms/step - loss: 3.7601

# 3. 用冻结的编码器训练分类模型
classifier = create_classifier(encoder, trainable=False)

history = classifier.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=num_epochs)

accuracy = classifier.evaluate(x_test, y_test)[1]
print(f"Test accuracy: {round(accuracy * 100, 2)}%")
# Epoch 1/50
# 189/189 [==============================] - 3s 16ms/step - loss: 0.3979 - sparse_categorical_accuracy: 0.8869
# Epoch 2/50
# 189/189 [==============================] - 3s 16ms/step - loss: 0.3422 - sparse_categorical_accuracy: 0.8959
# Epoch 3/50
# 189/189 [==============================] - 3s 16ms/step - loss: 0.3251 - sparse_categorical_accuracy: 0.9004
# Epoch 4/50
# 189/189 [==============================] - 3s 16ms/step - loss: 0.3313 - sparse_categorical_accuracy: 0.8963
# Epoch 5/50
# 189/189 [==============================] - 3s 16ms/step - loss: 0.3213 - sparse_categorical_accuracy: 0.9006
# Epoch 6/50
# 189/189 [==============================] - 3s 16ms/step - loss: 0.3221 - sparse_categorical_accuracy: 0.9001
# Epoch 7/50
# 189/189 [==============================] - 3s 16ms/step - loss: 0.3134 - sparse_categorical_accuracy: 0.9001
# Epoch 8/50
# 189/189 [==============================] - 3s 16ms/step - loss: 0.3245 - sparse_categorical_accuracy: 0.8978
# Epoch 9/50
# 189/189 [==============================] - 3s 16ms/step - loss: 0.3144 - sparse_categorical_accuracy: 0.9001
# Epoch 10/50
# 189/189 [==============================] - 3s 16ms/step - loss: 0.3191 - sparse_categorical_accuracy: 0.8984
# Epoch 11/50
# 189/189 [==============================] - 3s 16ms/step - loss: 0.3104 - sparse_categorical_accuracy: 0.9025
# Epoch 12/50
# 189/189 [==============================] - 3s 16ms/step - loss: 0.3261 - sparse_categorical_accuracy: 0.8958
# Epoch 13/50
# 189/189 [==============================] - 3s 16ms/step - loss: 0.3130 - sparse_categorical_accuracy: 0.9001
# Epoch 14/50
# 189/189 [==============================] - 3s 16ms/step - loss: 0.3147 - sparse_categorical_accuracy: 0.9003
# Epoch 15/50
# 189/189 [==============================] - 3s 16ms/step - loss: 0.3113 - sparse_categorical_accuracy: 0.9016
# Epoch 16/50
# 189/189 [==============================] - 3s 16ms/step - loss: 0.3114 - sparse_categorical_accuracy: 0.9008
# Epoch 17/50
# 189/189 [==============================] - 3s 16ms/step - loss: 0.3044 - sparse_categorical_accuracy: 0.9026
# Epoch 18/50
# 189/189 [==============================] - 3s 16ms/step - loss: 0.3142 - sparse_categorical_accuracy: 0.8987
# Epoch 19/50
# 189/189 [==============================] - 3s 16ms/step - loss: 0.3139 - sparse_categorical_accuracy: 0.9018
# Epoch 20/50
# 189/189 [==============================] - 3s 16ms/step - loss: 0.3199 - sparse_categorical_accuracy: 0.8987
# Epoch 21/50
# 189/189 [==============================] - 3s 16ms/step - loss: 0.3125 - sparse_categorical_accuracy: 0.8994
# Epoch 22/50
# 189/189 [==============================] - 3s 16ms/step - loss: 0.3291 - sparse_categorical_accuracy: 0.8967
# Epoch 23/50
# 189/189 [==============================] - 3s 16ms/step - loss: 0.3208 - sparse_categorical_accuracy: 0.8963
# Epoch 24/50
# 189/189 [==============================] - 3s 16ms/step - loss: 0.3065 - sparse_categorical_accuracy: 0.9041
# Epoch 25/50
# 189/189 [==============================] - 3s 16ms/step - loss: 0.3099 - sparse_categorical_accuracy: 0.9006
# Epoch 26/50
# 189/189 [==============================] - 3s 16ms/step - loss: 0.3181 - sparse_categorical_accuracy: 0.8986
# Epoch 27/50
# 189/189 [==============================] - 3s 16ms/step - loss: 0.3112 - sparse_categorical_accuracy: 0.9013
# Epoch 28/50
# 189/189 [==============================] - 3s 16ms/step - loss: 0.3136 - sparse_categorical_accuracy: 0.8996
# Epoch 29/50
# 189/189 [==============================] - 3s 16ms/step - loss: 0.3217 - sparse_categorical_accuracy: 0.8969
# Epoch 30/50
# 189/189 [==============================] - 3s 16ms/step - loss: 0.3161 - sparse_categorical_accuracy: 0.8998
# Epoch 31/50
# 189/189 [==============================] - 3s 16ms/step - loss: 0.3151 - sparse_categorical_accuracy: 0.8999
# Epoch 32/50
# 189/189 [==============================] - 3s 16ms/step - loss: 0.3092 - sparse_categorical_accuracy: 0.9009
# Epoch 33/50
# 189/189 [==============================] - 3s 16ms/step - loss: 0.3246 - sparse_categorical_accuracy: 0.8961
# Epoch 34/50
# 189/189 [==============================] - 3s 16ms/step - loss: 0.3143 - sparse_categorical_accuracy: 0.8995
# Epoch 35/50
# 189/189 [==============================] - 3s 16ms/step - loss: 0.3106 - sparse_categorical_accuracy: 0.9002
# Epoch 36/50
# 189/189 [==============================] - 3s 16ms/step - loss: 0.3210 - sparse_categorical_accuracy: 0.8980
# Epoch 37/50
# 189/189 [==============================] - 3s 16ms/step - loss: 0.3178 - sparse_categorical_accuracy: 0.9009
# Epoch 38/50
# 189/189 [==============================] - 3s 16ms/step - loss: 0.3064 - sparse_categorical_accuracy: 0.9032
# Epoch 39/50
# 189/189 [==============================] - 3s 16ms/step - loss: 0.3196 - sparse_categorical_accuracy: 0.8981
# Epoch 40/50
# 189/189 [==============================] - 3s 16ms/step - loss: 0.3177 - sparse_categorical_accuracy: 0.8988
# Epoch 41/50
# 189/189 [==============================] - 3s 16ms/step - loss: 0.3167 - sparse_categorical_accuracy: 0.8987
# Epoch 42/50
# 189/189 [==============================] - 3s 16ms/step - loss: 0.3110 - sparse_categorical_accuracy: 0.9014
# Epoch 43/50
# 189/189 [==============================] - 3s 16ms/step - loss: 0.3124 - sparse_categorical_accuracy: 0.9002
# Epoch 44/50
# 189/189 [==============================] - 3s 16ms/step - loss: 0.3128 - sparse_categorical_accuracy: 0.8999
# Epoch 45/50
# 189/189 [==============================] - 3s 16ms/step - loss: 0.3131 - sparse_categorical_accuracy: 0.8991
# Epoch 46/50
# 189/189 [==============================] - 3s 16ms/step - loss: 0.3149 - sparse_categorical_accuracy: 0.8992
# Epoch 47/50
# 189/189 [==============================] - 3s 16ms/step - loss: 0.3082 - sparse_categorical_accuracy: 0.9021
# Epoch 48/50
# 189/189 [==============================] - 3s 16ms/step - loss: 0.3223 - sparse_categorical_accuracy: 0.8959
# Epoch 49/50
# 189/189 [==============================] - 3s 16ms/step - loss: 0.3195 - sparse_categorical_accuracy: 0.8981
# Epoch 50/50
# 189/189 [==============================] - 3s 16ms/step - loss: 0.3240 - sparse_categorical_accuracy: 0.8962
# 313/313 [==============================] - 2s 7ms/step - loss: 0.7332 - sparse_categorical_accuracy: 0.8162
# Test accuracy: 81.62%
# We get to an improved test accuracy.


def main():
    pass


if __name__ == '__main__':
    main()
