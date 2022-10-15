#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense

from sklearn.datasets import load_breast_cancer

from keras.layers import Input,Embedding,LSTM,Dense
from keras.models import Model
from keras import backend as K

word_size = 128
nb_features = 10000
nb_classes = 10
encode_size = 64

input = Input(shape=(None,))
embedded = Embedding(nb_features,word_size)(input)
encoder = LSTM(encode_size)(embedded)
predict = Dense(nb_classes, activation='softmax')(encoder)

def mycrossentropy(y_true, y_pred, e=0.1):
    # 第一项就是普通的交叉熵，第二项中，先通过K.ones_like(y_pred)/nb_classes构造了一个均匀分布，然后算y_pred与均匀分布的交叉熵。
    loss1 = K.categorical_crossentropy(y_true, y_pred)
    loss2 = K.categorical_crossentropy(K.ones_like(y_pred)/nb_classes, y_pred)
    return (1-e)*loss1 + e*loss2

model = Model(inputs=input, outputs=predict)
model.compile(optimizer='adam', loss=mycrossentropy)


y_true = tf.constant([1., 0., 0., 0., 1., 0., 0., 0., 1.], shape=[3, 3])
y_pred = tf.constant([.9, .05, .05, .05, .89, .06, .05, .01, .94], shape=[3, 3])
mycrossentropy(y_true, y_pred, e=0.1)

def GHMC_Loss(Bins=10, momentum=0, batch_size=100):
    global edges, mmt, bins, shape  ##
    shape = batch_size
    bins = Bins
    edges = [float(x) / bins for x in range(bins + 1)]
    edges[-1] += 1e-6
    mmt = momentum
    if momentum > 0:
        acc_sum = [0.0 for _ in range(bins)]

    def GHMC_Loss_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        weights = tf.numpy_function(GHMC_loss_pyfunc, [y_true, y_pred], tf.float32)

        weights = tf.reshape(weights, [shape])
        return tf.nn.weighted_cross_entropy_with_logits(labels=y_true, logits=y_pred, pos_weight=weights)

    return GHMC_Loss_fixed


def GHMC_loss_pyfunc(y_true, y_pred, momentum=0):
    """计算每个样本的权重
    weight = 总样本数/(当前类别数*总区间数)
    如样本x,pred=0.01, label=1, 总样本数为569，0-0.1区间有357个样本，故而其权重为: 569/(357*2)=0.7969187675070029
    同理样本y,pred=0.01, label=1, 总样本数为569，0-0.1区间有212个样本，故而其权重为: 569/(212*2)=1.3419811320754718
    总之，样本区间样本数越少，其权重越大；
    """
    weights = np.zeros(shape)
    g = np.abs(y_true - y_pred)
    tot = len(y_pred)
    n = 0
    for i in range(bins):
        inds = (g >= edges[i]) & (g < edges[i + 1])
        num_in_bin = inds.sum().item()
        if num_in_bin > 0:
            if momentum > 0:
                acc_sum[i] = momentum * acc_sum[i] + (1 - mmt) * num_in_bin
                weights[inds.flatten()] = tot / acc_sum[i]
            else:
                weights[inds.flatten()] = tot / num_in_bin  ##N/GD(gi)
            n += 1
    if n > 0:
        weights = weights / n

    return weights.astype('float32')


def main():

    X = load_breast_cancer().data
    y = load_breast_cancer().target
    model = Sequential()
    model.add(Dense(60, input_dim=30, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss=GHMC_Loss(Bins=10, momentum=0, batch_size=569), optimizer='adam', metrics=['accuracy'])
    model.fit(X, y, epochs=2, batch_size=569)


if __name__ == '__main__':
    main()
