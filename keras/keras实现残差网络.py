#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from keras import layers
import keras
import numpy as np

inputs = keras.Input(shape=(32, 32, 3), name="img")
x = layers.Conv2D(32, 3, activation="relu")(inputs)
x = layers.Conv2D(64, 3, activation="relu")(x)
block_1_output = layers.MaxPooling2D(3)(x)

x = layers.Conv2D(64, 3, activation="relu", padding="same")(block_1_output)
x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
block_2_output = layers.add([x, block_1_output])

x = layers.Conv2D(64, 3, activation="relu", padding="same")(block_2_output)
x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
block_3_output = layers.add([x, block_2_output])

x = layers.Conv2D(64, 3, activation="relu")(block_3_output)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(10)(x)

model = keras.Model(inputs, outputs, name="toy_resnet")
model.summary()

# 绘制模型
# keras.utils.plot_model(model, "./result/images/mini_resnet.png", show_shapes=True)

# # 训练模型
# (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# x_train = x_train.astype("float32") / 255.0
# x_test = x_test.astype("float32") / 255.0
# y_train = keras.utils.to_categorical(y_train, 10)
# y_test = keras.utils.to_categorical(y_test, 10)

# model.compile(
#  optimizer=keras.optimizers.RMSprop(1e-3),
#  loss="categorical_crossentropy",
#  metrics=["accuracy"],
# )
# # We restrict the data to the first 1000 samples so as to limit execution time
# # on Colab. Try to train on the entire dataset until convergence!
# model.fit(x_train[:1000], y_train[:1000], batch_size=64, epochs=1, validation_split=0.2)

# __________________________________________________________________________________________________
# Layer (type)                    Output Shape         Param #     Connected to
# ==================================================================================================
# img (InputLayer)                (None, 32, 32, 3)    0
# __________________________________________________________________________________________________
# conv2d_1 (Conv2D)               (None, 30, 30, 32)   896         img[0][0]
# __________________________________________________________________________________________________
# conv2d_2 (Conv2D)               (None, 28, 28, 64)   18496       conv2d_1[0][0]
# __________________________________________________________________________________________________
# max_pooling2d_1 (MaxPooling2D)  (None, 9, 9, 64)     0           conv2d_2[0][0]
# __________________________________________________________________________________________________
# conv2d_3 (Conv2D)               (None, 9, 9, 64)     36928       max_pooling2d_1[0][0]
# __________________________________________________________________________________________________
# conv2d_4 (Conv2D)               (None, 9, 9, 64)     36928       conv2d_3[0][0]
# __________________________________________________________________________________________________
# add_1 (Add)                     (None, 9, 9, 64)     0           conv2d_4[0][0]
#                                                                  max_pooling2d_1[0][0]
# __________________________________________________________________________________________________
# conv2d_5 (Conv2D)               (None, 9, 9, 64)     36928       add_1[0][0]
# __________________________________________________________________________________________________
# conv2d_6 (Conv2D)               (None, 9, 9, 64)     36928       conv2d_5[0][0]
# __________________________________________________________________________________________________
# add_2 (Add)                     (None, 9, 9, 64)     0           conv2d_6[0][0]
#                                                                  add_1[0][0]
# __________________________________________________________________________________________________
# conv2d_7 (Conv2D)               (None, 7, 7, 64)     36928       add_2[0][0]
# __________________________________________________________________________________________________
# global_average_pooling2d_1 (Glo (None, 64)           0           conv2d_7[0][0]
# __________________________________________________________________________________________________
# dense_1 (Dense)                 (None, 256)          16640       global_average_pooling2d_1[0][0]
# __________________________________________________________________________________________________
# dropout_1 (Dropout)             (None, 256)          0           dense_1[0][0]
# __________________________________________________________________________________________________
# dense_2 (Dense)                 (None, 10)           2570        dropout_1[0][0]
# ==================================================================================================
# Total params: 223,242
# Trainable params: 223,242
# Non-trainable params: 0
# __________________________________________________________________________________________________

# 展示网络结构：

import pydot
from keras.utils import plot_model
from IPython.display import Image
plot_model(model, show_shapes=True, show_layer_names=True, to_file="./result/images/mini_resnet.png")
Image("./result/images/mini_resnet.png")

def main():
    pass


if __name__ == '__main__':
    main()