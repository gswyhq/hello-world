#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from keras.datasets import boston_housing
from keras import models
from keras import layers

(X_train, y_train), (X_test, y_test) = boston_housing.load_data()#加载数据

#对数据进行标准化预处理，方便神经网络更好的学习
mean = X_train.mean(axis=0)
X_train -= mean
std = X_train.std(axis=0)
X_train /= std
X_test -= mean
X_test /= std

#构建神经网络模型
def build_model():
    #这里使用Sequential模型
    model = models.Sequential()
    #进行层的搭建，注意第二层往后没有输入形状(input_shape)，它可以自动推导出输入的形状等于上一层输出的形状
    model.add(layers.Dense(64, activation='relu',input_shape=(X_train.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    #编译网络
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    model.summary()
    return model

num_epochs = 5
model = build_model()
model.fit(X_train, y_train,epochs=num_epochs, batch_size=1, verbose=0)
predicts = model.predict(X_test)
print(predicts)

def main():
    pass


if __name__ == '__main__':
    main()