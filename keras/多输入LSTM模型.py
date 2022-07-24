#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# !/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np

import tensorflow as tf
import random as rn

# random seeds for stochastic parts of neural network
np.random.seed(10)
from tensorflow import set_random_seed

set_random_seed(15)

from keras.models import Model
from keras.layers import Input, Dense, Concatenate, Reshape, Dropout, TimeDistributed
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
from keras.callbacks import TensorBoard
import os

np.random.seed(42)

rn.seed(12345)

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

from keras import backend as K

tf.set_random_seed(1234)

sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

# In[3]:


import pandas as pd
import os, time
import numpy as np
import requests
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix, average_precision_score, \
    precision_recall_curve
import matplotlib.pyplot as plt
from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
from imblearn.under_sampling import RandomUnderSampler, NearMiss, TomekLinks, EditedNearestNeighbours
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn import preprocessing
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
from datetime import datetime
import pickle
import json
from tqdm import tqdm
import time
from glob import glob
import numpy as np
import os, time, sys, argparse
import tensorflow as tf
from tensorflow import keras
from tensorflow.contrib.rnn import LSTMCell
import logging, sys, argparse
import random
import pandas as pd
import pickle
import math

from tqdm import tqdm
import tensorflow as tf
from tensorflow.contrib.crf import viterbi_decode
from tensorflow.python.platform import gfile
from tensorflow.python.framework import graph_util

# In[4]:


print(tf.__version__, keras.__version__)
# 1.14.0 2.2.4-tf

# In[5]:


def Precision(y_true, y_pred):
    """精确率"""
    tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))  # true positives
    pp = K.sum(K.round(K.clip(y_pred, 0, 1)))  # predicted positives
    precision = tp / (pp + K.epsilon())
    return precision


def Recall(y_true, y_pred):
    """召回率"""
    tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))  # true positives
    pp = K.sum(K.round(K.clip(y_true, 0, 1)))  # possible positives
    recall = tp / (pp + K.epsilon())
    return recall


def F1(y_true, y_pred):
    """F1-score"""
    precision = Precision(y_true, y_pred)
    recall = Recall(y_true, y_pred)
    f1 = 2 * ((precision * recall) / (precision + recall + K.epsilon()))
    return f1


def wubao(y_true, y_pred):
    """误报率：误报/所有预警"""
    precision = Precision(y_true, y_pred)
    return 1 - precision


def loubao(y_true, y_pred):
    """漏报率：漏报/所有升级"""
    recall = Recall(y_true, y_pred)
    return 1 - recall


tf.compat.v1.disable_eager_execution()
with tf.compat.v1.Session() as sess:
    print(sess.run(loubao(tf.constant(np.array([0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0]), dtype='float'),
                          tf.constant(np.array(
                              [0.0, 0.7, 0.0446, 0.0, 0.06668, 0.6279, 0.8957, 0.2, 1.0, 0.9, 0.61, 0.0, 0.0768, 0.109,
                               0.8]), dtype='float'))))


# [[5 2]
#  [3 5]]
# 误报率：0.286
# 漏报率：0.375
# 预警率：0.467


# In[7]:


def build_model3():
    input_1 = Input(shape=(6, 44,), name="input_1")
    input_2 = Input(shape=(6, 1,), name="input_2")
    embedding = Embedding(737, 32, input_length=(6, 44))(input_1)  # ?, 44,32
    print(embedding)
    time_embedding = TimeDistributed(Dense(32), input_shape=(6, 44, 32))(embedding)
    print(time_embedding)
    embedding = Reshape(target_shape=(6, 44 * 32,))(time_embedding)  # ?,1408
    embedding = Dropout(0.2)(embedding)

    time_input_2 = TimeDistributed(Dense(1), input_shape=(6, 1))(input_2)
    concat = Concatenate()([embedding, time_input_2])  # ?,1409
    print(concat)
    # LSTMs的输入必须是三维的（三维的结构是[样本批大小，滑窗大小，特征数量]
    #     lstm1 = LSTM(256, return_sequences=True)(concat)
    #     lstm1 = Dropout(0.2)(lstm1)
    #     concat = Reshape(target_shape=(1049, 1))(concat)
    #     print(concat)
    lstm2 = LSTM(256, input_shape=(6, 1049), return_sequences=False)(concat)
    lstm2 = Dropout(0.2)(lstm2)

    dense1 = Dense(32, kernel_initializer="uniform", activation='relu')(lstm2)
    dense2 = Dense(1, kernel_initializer="uniform", activation='sigmoid', name="output")(dense1)
    # model = load_model('my_LSTM_stock_model1000.h5')
    inputs = [input_1, input_2]
    outputs = [dense2]
    model = Model(inputs, outputs)
    #     adam = keras.optimizers.Adam(decay=0.2)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', wubao, loubao])
    model.summary()
    return model

# Tensor("embedding_1/embedding_lookup/Identity:0", shape=(?, 6, 44, 32), dtype=float32)
# Tensor("time_distributed_1/Reshape_5:0", shape=(?, 6, 44, 32), dtype=float32)
# Tensor("concatenate_1/concat:0", shape=(?, 6, 1409), dtype=float32)
# __________________________________________________________________________________________________
# Layer (type)                    Output Shape         Param #     Connected to
# ==================================================================================================
# input_1 (InputLayer)            (None, 6, 44)        0
# __________________________________________________________________________________________________
# embedding_1 (Embedding)         (None, 6, 44, 32)    23584       input_1[0][0]
# __________________________________________________________________________________________________
# time_distributed_1 (TimeDistrib (None, 6, 44, 32)    1056        embedding_1[0][0]
# __________________________________________________________________________________________________
# reshape_1 (Reshape)             (None, 6, 1408)      0           time_distributed_1[0][0]
# __________________________________________________________________________________________________
# input_2 (InputLayer)            (None, 6, 1)         0
# __________________________________________________________________________________________________
# dropout_1 (Dropout)             (None, 6, 1408)      0           reshape_1[0][0]
# __________________________________________________________________________________________________
# time_distributed_2 (TimeDistrib (None, 6, 1)         2           input_2[0][0]
# __________________________________________________________________________________________________
# concatenate_1 (Concatenate)     (None, 6, 1409)      0           dropout_1[0][0]
#                                                                  time_distributed_2[0][0]
# __________________________________________________________________________________________________
# lstm_1 (LSTM)                   (None, 256)          1705984     concatenate_1[0][0]
# __________________________________________________________________________________________________
# dropout_2 (Dropout)             (None, 256)          0           lstm_1[0][0]
# __________________________________________________________________________________________________
# dense_3 (Dense)                 (None, 32)           8224        dropout_2[0][0]
# __________________________________________________________________________________________________
# output (Dense)                  (None, 1)            33          dense_3[0][0]
# ==================================================================================================
# Total params: 1,738,883
# Trainable params: 1,738,883
# Non-trainable params: 0
# __________________________________________________________________________________________________

keras.backend.clear_session()
model = build_model3()

# In[8]:


rng = np.random.RandomState(123)
input_1 = rng.randint(736, size=(512, 6, 44))
input_2 = rng.randint(100, size=(512, 6, 1))
input_y = rng.randint(2, size=(512, 1))

# In[10]:


config = tf.ConfigProto(
    #     device_count={'GPU': 1},
    intra_op_parallelism_threads=1,
    allow_soft_placement=True
)

config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.6

session = tf.Session(config=config)

keras.backend.set_session(session)

time_str = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))
model_save_path = "./result/{}".format(time_str)
if not os.path.isdir(model_save_path):
    os.mkdir(model_save_path)
log_dir = os.path.join(model_save_path, 'logs')
batch_size = 16
early_stopping = EarlyStopping(monitor='val_loss', patience=3)  # 连续 patience 次，val_loss 都没有得到改善则退出训练。

model_checkpoint = ModelCheckpoint(filepath=os.path.join(model_save_path,
                                                         'classify-{epoch:02d}-{acc:.4f}-{wubao:.4f}-{loubao:.4f}.hdf5'),
                                   save_best_only=True,
                                   save_weights_only=False)

tb = TensorBoard(log_dir=log_dir,  # log 目录
                 histogram_freq=0,  # 按照何等频率（epoch）来计算直方图，0为不计算
                 batch_size=batch_size,  # 用多大量的数据计算直方图
                 write_graph=True,  # 是否存储网络结构图
                 write_grads=False,  # 是否可视化梯度直方图
                 write_images=False,  # 是否可视化参数
                 embeddings_freq=0,
                 embeddings_layer_names=None,
                 embeddings_metadata=None)

with session.as_default():
    with session.graph.as_default():
        model.fit(x={"input_1": input_1,
                     "input_2": input_2},
                  y=input_y,
                  #                   validation_data=(validation_x, validation_y),
                  epochs=3,
                  batch_size=batch_size,
                  verbose=1,
                  shuffle=True,
                  callbacks=[early_stopping, model_checkpoint, tb]
                  )

# Epoch 1/3
# 32/32 [==============================] - 1s 35ms/step - loss: 0.6929 - acc: 0.5938 - wubao: 0.4148 - loubao: 0.3125
# Epoch 2/3
# 16/32 [==============>...............] - ETA: 0s - loss: 0.6928 - acc: 0.5625 - wubao: 0.4167 - loubao: 0.2222
# 32/32 [==============================] - 0s 7ms/step - loss: 0.6930 - acc: 0.5312 - wubao: 0.4750 - loubao: 0.1111
# Epoch 3/3
# 32/32 [==============================] - 0s 7ms/step - loss: 0.6911 - acc: 0.5000 - wubao: 0.5000 - loubao: 0.0000e+00

def main():
    pass


if __name__ == '__main__':
    main()