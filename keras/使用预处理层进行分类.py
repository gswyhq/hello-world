#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import pathlib
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers.experimental import preprocessing as tf_preprocessing
from tensorflow.keras.layers.experimental.preprocessing import Normalization
import numpy as np
import os, time, sys, argparse
import tensorflow as tf
from tensorflow import keras
# from tensorflow.contrib.rnn import LSTMCell
import logging, sys, argparse
import random
import pandas as pd
import pickle
import math
from tqdm import tqdm
import random as rn
import pydot
from keras.utils.vis_utils import plot_model
# from adjustText import adjust_text
from keras import regularizers
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization # tf2.60
from IPython.display import Image
from keras.models import Model
from keras.layers import Input, Dense, Concatenate, Reshape, Dropout, TimeDistributed
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
from keras.callbacks import TensorBoard
from keras.layers import Dense,LSTM,Bidirectional
import os
import keras
from keras.models import load_model
from keras import backend as K
# from tensorflow.python.keras import backend as K        #from keras import backend as K
from tensorflow.python.framework import graph_io
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from sklearn.manifold import TSNE
import os, time
import requests
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix, average_precision_score, precision_recall_curve
import matplotlib.pyplot as plt
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei'] #指定默认字体   
mpl.rcParams['axes.unicode_minus'] = False #解决保存图像是负号'-'显示为方块的问题
from imblearn.under_sampling import RandomUnderSampler,NearMiss,TomekLinks,EditedNearestNeighbours
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn import preprocessing
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier, XGBRegressor
from datetime import datetime
import json
import time
from glob import glob
# from tensorflow.contrib.crf import viterbi_decode
from tensorflow.python.platform import gfile
from tensorflow.python.framework import graph_util
from keras.utils.np_utils import to_categorical
from scipy import stats
import os
import keras
from keras.models import load_model
import tensorflow as tf
from keras import backend as K
# from tensorflow.python.keras import backend as K        #from keras import backend as K
from tensorflow.python.framework import graph_io
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

# tf.__version__
# Out[25]: '2.6.0'
# keras.__version__
# Out[26]: '2.6.0'

train = pd.DataFrame([['投诉', 4, 0.23, 1], ['投诉', 6, 0.23, 1], ['投诉', 6, 0.23, 1], ['投诉', 6, 0.23, 1], ['投诉', 6, 0.23, 1], ['投诉', 3, 0.23, 1], ['投诉', 3, 0.23, 1], ['投诉', 3, 0.23, 1], ['投诉', 3, 0.23, 1], ['投诉', 3, 0.21, 1], ['投诉', 3, 0.26, 1], ['投诉', 4, 0.27, 1], ['投诉', 4, 0.23, 1], ['投诉', 4, 0.23, 1], ['投诉', 4, 0.23, 1], ['投诉', 4, 0.23, 1], ['投诉', 4, 0.23, 1], ['投诉', 4, 0.23, 1], ['投诉', 4, 0.23, 1], ['投诉', 7, 0.23, 1], ['服务', 10, 0.28, 1], ['服务', 10, 0.27, 1], ['服务', 10, 0.23, 1], ['服务', 10, 0.22, 1], ['服务', 10, 0.23, 0], ['服务', 10, 0.29, 1], ['服务', 10, 0.23, 0], ['服务', 10, 0.35, 0], ['服务', 10, 0.23, 0], ['服务', 10, 0.23, 0]],
                     columns=['类别', '月份', '年化利率', 'target'])
val = pd.DataFrame([['投诉', 3, 0.23, 1], ['投诉', 3, 0.23, 1], ['投诉', 3, 0.23, 1], ['服务', 10, 0.29, 0], ['服务', 10, 0.24, 1]],
                     columns=['类别', '月份', '年化利率', 'target'])

def build_model2():
    all_inputs2 = []
    encoded_features2 = []
    keras.backend.clear_session()
    # Numeric features.

    numeric_col = tf.keras.Input(shape=(1,), name='年化利率')
    # normalization_layer = get_normalization_layer('年化利率', train_ds)
    normalization_layer = tf_preprocessing.Normalization(mean=train['年化利率'].values.mean(), variance=train['年化利率'].values.var())  # 年化利率的均值和方差
    encoded_numeric_col = normalization_layer(numeric_col)
    all_inputs2.append(numeric_col)
    encoded_features2.append(encoded_numeric_col)

    # Categorical features encoded as integers.
    numeric_col = tf.keras.Input(shape=(1,), name='月份', dtype='int64')
    # encoding_layer = get_category_encoding_layer('月份', train_ds, dtype='int64',
    #                                              max_tokens=None)
    # encoded_numeric_col = encoding_layer(numeric_col)
    encoding_layer1 = tf_preprocessing.IntegerLookup(vocabulary=[ 10, 4, 3, 9, 8, 7, 12, 1, 5, 6, 11, 2], dtype='int64')  # 月份 词汇的查找层
    print(encoding_layer1.get_vocabulary())
    encoder_layer = tf_preprocessing.CategoryEncoding(num_tokens=encoding_layer1.vocabulary_size())
    # lambda_layer = tf.keras.layers.Lambda(lambda feature: encoder(encoding_layer(feature)))

    encoded_numeric_col1 = encoding_layer1(numeric_col)
    encoded_numeric_col = encoder_layer(encoded_numeric_col1)
    all_inputs2.append(numeric_col)
    encoded_features2.append(encoded_numeric_col)

    # Categorical features encoded as string.
    categorical_col = tf.keras.Input(shape=(1,), name='类别', dtype='string')
    # encoding_layer = get_category_encoding_layer('类别', train_ds, dtype='string',
    #                                            max_tokens=None)
    # encoded_categorical_col = encoding_layer(categorical_col)
    encoding_layer2 = tf_preprocessing.StringLookup(vocabulary=['咨询', '投诉', '服务'], dtype='string')  # 类别 词汇的查找层
    print(encoding_layer2.get_vocabulary())
    encoder_layer = tf_preprocessing.CategoryEncoding(num_tokens=encoding_layer2.vocabulary_size())
    # lambda_layer = tf.keras.layers.Lambda(lambda feature: encoder(encoding_layer(feature)))

    categorical_col1 = encoding_layer2(categorical_col)
    encoded_categorical_col = encoder_layer(categorical_col1)

    all_inputs2.append(categorical_col)
    encoded_features2.append(encoded_categorical_col)

    # 创建、编译并训练模型
    # 接下来，您可以创建端到端模型。


    all_features = Concatenate()(encoded_features2)
    x = tf.keras.layers.Dense(32, activation="relu")(all_features)
    x = tf.keras.layers.Dropout(0.5)(x)
    output = tf.keras.layers.Dense(1)(x)
    model2 = tf.keras.Model(all_inputs2, output)
    model2.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=["accuracy"])
    model2.summary()
    return model2
model2 = build_model2()

model2.fit({"年化利率": train['年化利率'].values,
           "月份": train['月份'].values,
           "类别": train['类别'].values},  train['target'].values
           , epochs=2, validation_data=({"年化利率": val['年化利率'].values,
           "月份": val['月份'].values,
           "类别": val['类别'].values, }, val['target'].values))

sample = {'年化利率': 0.24, '月份': 9, '类别': "投诉"}
input_dict = {name: tf.convert_to_tensor([value]) for name, value in sample.items()}
predictions1 = model2.predict(input_dict)
print('预测结果:', tf.nn.sigmoid(predictions1[0]))

# 上模型，保存的pb模型，加载PB模型的时候，可能出错
# /usr/local/lib/python3.8/dist-packages/tensorflow/python/framework/dtypes.py in as_numpy_dtype(self)
#      87   def as_numpy_dtype(self):
#      88     """Returns a Python `type` object based on this `DType`."""
# ---> 89     return _TF_TO_NP[self._type_enum]
#      90
#      91   @property
#
# KeyError: 20

# 故而将预处理层剥离出去

###################################################剥离出预处理层#####################################################################

encoding_layer1 = tf_preprocessing.IntegerLookup(vocabulary=[ 10, 4, 3, 9, 8, 7, 12, 1, 5, 6, 11, 2], dtype='int64')  # 月份 词汇的查找层
print(encoding_layer1.get_vocabulary())
encoder_layer1 = tf_preprocessing.CategoryEncoding(num_tokens=encoding_layer1.vocabulary_size())

encoding_layer2 = tf_preprocessing.StringLookup(vocabulary=['咨询', '投诉', '服务'], dtype='string')  # 类别 词汇的查找层
print(encoding_layer2.get_vocabulary())
encoder_layer2 = tf_preprocessing.CategoryEncoding(num_tokens=encoding_layer2.vocabulary_size())

def build_model3():
    all_inputs2 = []
    encoded_features2 = []
    keras.backend.clear_session()
    # Numeric features.

    numeric_col = tf.keras.Input(shape=(1,), name='年化利率')
    # normalization_layer = get_normalization_layer('年化利率', train_ds)
    normalization_layer = tf_preprocessing.Normalization(mean=0.27292730635180307,
                                                         variance=0.0038312310055545577)  # 年化利率的均值和方差
    encoded_numeric_col = normalization_layer(numeric_col)
    all_inputs2.append(numeric_col)
    encoded_features2.append(encoded_numeric_col)

    # Categorical features encoded as integers.
    numeric_col = tf.keras.Input(shape=(13,), name='月份', dtype='float32')
    # encoding_layer = get_category_encoding_layer('月份', train_ds, dtype='int64',
    #                                              max_tokens=None)

    all_inputs2.append(numeric_col)
    encoded_features2.append(numeric_col)

    # Categorical features encoded as string.
    categorical_col = tf.keras.Input(shape=(4,), name='类别', dtype='float32')
    # encoding_layer = get_category_encoding_layer('类别', train_ds, dtype='string',
    #                                            max_tokens=None)
    # encoded_categorical_col = encoding_layer(categorical_col)

    all_inputs2.append(categorical_col)
    encoded_features2.append(categorical_col)

    # 创建、编译并训练模型
    # 接下来，您可以创建端到端模型。

    all_features = Concatenate()(encoded_features2)
    x = tf.keras.layers.Dense(32, activation="relu")(all_features)
    x = tf.keras.layers.Dropout(0.5)(x)
    output = tf.keras.layers.Dense(1)(x)
    model3 = tf.keras.Model(all_inputs2, output)
    model3.compile(optimizer='adam',
                   loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                   metrics=["accuracy"])
    model3.summary()

    return model3

model3 = build_model3()
model3.fit({"年化利率": train['年化利率'].values,
           "月份": encoder_layer1(encoding_layer1(np.reshape(train['月份'].values, (train.shape[0], 1)))),
           "类别": encoder_layer2(encoding_layer2(np.reshape(train['类别'].values, (train.shape[0], 1)))),}, train['target'].values
           , epochs=2, validation_data=({"年化利率": val['年化利率'].values,
           "月份": encoder_layer1(encoding_layer1(np.reshape(val['月份'].values, (val.shape[0], 1)))),
           "类别": encoder_layer2(encoding_layer2(np.reshape(val['类别'].values, (val.shape[0], 1)))),}, val['target'].values))

sample = {'年化利率': 0.24, '月份': encoder_layer1(encoding_layer1(np.array(9))), '类别': encoder_layer2(encoding_layer2(np.array("投诉")))}
input_dict = {name: tf.convert_to_tensor([value]) for name, value in sample.items()}
predictions1 = model3.predict(input_dict)
print("模型预测结果：", tf.nn.sigmoid(predictions1[0]))

def main():
    pass


if __name__ == '__main__':
    main()
