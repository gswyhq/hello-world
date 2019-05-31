#!/usr/bin/python3
# coding: utf-8

# https://blog.csdn.net/wcy23580/article/details/84990923

# https://github.com/bradleypallen/keras-quora-question-pairs.git

# 文章目录
# 1.
# input
# encoding
# 1.1
# 原理
# 1.2
# 实现
# 2.
# local
# inference
# modeling
# 2.1
# 原理
# 2.2
# 实现
# 3.
# inference
# composition
# 3.1
# 原理
# 3.2
# 实现
#
# ESIM
# 原理笔记见：论文笔记 & 翻译——Enhanced
# LSTM
# for Natural Language Inference(ESIM)
# ESIM主要分为三部分：input
# encoding，local
# inference
# modeling
# 和
# inference
# composition。如上图所示，ESIM
# 是左边一部分, 如下图所示
#
# 三部分简要代码如下：
#
# 1.
# input
# encoding
# 1.1
# 原理
#
# 1.2
# 实现
from keras.layers import Embedding, LSTM, Dense, Input, Bidirectional
from keras.layers.merge import Dot, Multiply, Concatenate, Subtract
from keras.layers.advanced_activations import Softmax
from keras.layers.core import Lambda, Dense, Dropout
from keras import backend as K
from keras.layers.wrappers import Bidirectional
from keras.layers.recurrent import LSTM
from keras.layers import Input, Dense
from keras.models import Model

i1 = Input(shape=(SentenceLen,), dtype='float32')
i2 = Input(shape=(SentenceLen,), dtype='float32')

x1 = Embedding([CONFIG])(i1)
x2 = Embedding([CONFIG])(i2)

x1 = Bidirectional(LSTM(300, return_sequences=True))(x1)
x2 = Bidirectional(LSTM(300, return_sequences=True))(x2)


# 2.
# local
# inference
# modeling
# 2.1
# 原理
#
# 2.2
# 实现
e = Dot(axes=2)([x1, x2])
e1 = Softmax(axis=2)(e)
e2 = Softmax(axis=1)(e)
e1 = Lambda(K.expand_dims, arguments={'axis': 3})(e1)
e2 = Lambda(K.expand_dims, arguments={'axis': 3})(e2)

_x1 = Lambda(K.expand_dims, arguments={'axis': 1})(x2)
_x1 = Multiply()([e1, _x1])
_x1 = Lambda(K.sum, arguments={'axis': 2})(_x1)
_x2 = Lambda(K.expand_dims, arguments={'axis': 2})(x1)
_x2 = Multiply()([e2, _x2])
_x2 = Lambda(K.sum, arguments={'axis': 1})(_x2)


# 3.
# inference
# composition
# 3.1
# 原理
#
#
#
# 3.2
# 实现
m1 = Concatenate()([x1, _x1, Subtract()([x1, _x1]), Multiply()([x1, _x1])])
m2 = Concatenate()([x2, _x2, Subtract()([x2, _x2]), Multiply()([x2, _x2])])

y1 = Bidirectional(LSTM(300, return_sequences=True))(m1)
y2 = Bidirectional(LSTM(300, return_sequences=True))(m2)

mx1 = Lambda(K.max, arguments={'axis': 1})(y1)
av1 = Lambda(K.mean, arguments={'axis': 1})(y1)
mx2 = Lambda(K.max, arguments={'axis': 1})(y2)
av2 = Lambda(K.mean, arguments={'axis': 1})(y2)

y = Concatenate()([av1, mx1, av2, mx2])
y = Dense(1024, activation='tanh')(y)
y = Dropout(0.5)(y)
y = Dense(1024, activation='tanh')(y)
y = Dropout(0.5)(y)
y = Dense(2, activation='softmax')(y)


# 快速开始函数式(Functional)模型
# https://keras-cn.readthedocs.io/en/latest/getting_started/functional_API/

def main():
    pass


if __name__ == '__main__':
    main()