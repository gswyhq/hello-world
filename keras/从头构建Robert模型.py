#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import numpy as np
import math
import pandas as pd
import keras
from keras.layers import Input, Lambda, Dense
from keras.models import Model, load_model
import keras.backend as K
import tensorflow as tf
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
from tensorflow.keras.utils import Sequence
from keras.callbacks import History
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from tqdm import tqdm
from keras.optimizer_v2.adam import Adam
from keras.models import load_model
import matplotlib.pyplot as plt
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei'] #指定默认字体 
mpl.rcParams['axes.unicode_minus'] = False #解决保存图像是负号'-'显示为方块的问题
import time

import pydot
from keras.utils.vis_utils import plot_model
from IPython.display import Image

from keras.engine.input_layer import InputLayer
from keras_bert.layers.embedding import TokenEmbedding
from keras.layers.embeddings import Embedding
from keras.layers.merge import Add
from keras_pos_embd.pos_embd import PositionEmbedding
from keras.layers.core import Dropout
from keras_layer_normalization.layer_normalization import LayerNormalization
from keras_multi_head.multi_head_attention import MultiHeadAttention
from keras_position_wise_feed_forward.feed_forward import FeedForward


config_path = r'D:\Users\{}\data\RoBERTa-tiny-clue\bert_config.json'.format(os.getenv("USERNAME"))
checkpoint_path = r'D:\Users\{}\data\RoBERTa-tiny-clue\bert_model.ckpt'.format(os.getenv("USERNAME"))
dict_path = r'D:\Users\{}\data\RoBERTa-tiny-clue\vocab.txt'.format(os.getenv("USERNAME"))

# 加载模型并展示
bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=None)
bert_model.summary()

# 展示模型结构
plot_model(bert_model, show_shapes=True, show_layer_names=True, to_file="./result/images/{}.png".format(time_str))
Image("./result/images/{}.png".format(time_str))  # 终端显示图片

# 输出模型每一层及其导入路径
for layer in bert_model.layers:
    print(layer.name, layer, layer.input)

from keras.engine.input_layer import InputLayer
from keras_bert.layers.embedding import TokenEmbedding
from keras.layers.embeddings import Embedding
from keras.layers.merge import Add
from keras_pos_embd.pos_embd import PositionEmbedding
from keras.layers.core import Dropout
from keras_layer_normalization.layer_normalization import LayerNormalization
from keras_multi_head.multi_head_attention import MultiHeadAttention
from keras_position_wise_feed_forward.feed_forward import FeedForward


def build_bert_model(output_layer_num=4, metrics=None):
    Input_Token = Input(shape=(None,), name="Input-Token")
    Input_Segment = Input(shape=(None,), name="Input-Segment")

    Embedding_Token = TokenEmbedding(input_dim=8021, output_dim=312, name="Embedding-Token")(Input_Token)
    Embedding_Segment = Embedding(input_dim=2, output_dim=312, name="Embedding-Segment")(Input_Segment)

    Embedding_Token_Segment = Add(name="Embedding-Token-Segment")([Embedding_Token[0], Embedding_Segment])
    Embedding_Position = PositionEmbedding(input_dim=512, output_dim=312, mode='add',
                                           embeddings_regularizer=None,
                                           embeddings_constraint=None,
                                           mask_zero=False,
                                           name="Embedding-Position")(Embedding_Token_Segment)
    Embedding_Dropout = Dropout(rate=0.1, name="Embedding-Dropout")(Embedding_Position)
    _Norm = LayerNormalization(name="Embedding-Norm")(Embedding_Dropout)
    for layer_index in range(1, output_layer_num + 1):
        MultiHeadSelfAttention = MultiHeadAttention(head_num=12,
                                                    name="Encoder-{}-MultiHeadSelfAttention".format(layer_index))(_Norm)
        MultiHeadSelfAttention_Dropout = Dropout(rate=0.1,
                                                 name="Encoder-{}-MultiHeadSelfAttention-Dropout".format(layer_index))(
            MultiHeadSelfAttention)
        MultiHeadSelfAttention_Add = Add(name="Encoder-{}-MultiHeadSelfAttention-Add".format(layer_index))(
            [_Norm, MultiHeadSelfAttention_Dropout])
        MultiHeadSelfAttention_Norm = LayerNormalization(
            name="Encoder-{}-MultiHeadSelfAttention-Norm".format(layer_index))(MultiHeadSelfAttention_Add)
        feedForward = FeedForward(units=1248, name="Encoder-{}-FeedForward".format(layer_index))(
            MultiHeadSelfAttention_Norm)
        FeedForward_Dropout = Dropout(rate=0.1, name="Encoder-{}-FeedForward-Dropout".format(layer_index))(feedForward)
        FeedForward_Add = Add(name="Encoder-{}-FeedForward-Add".format(layer_index))(
            [MultiHeadSelfAttention_Norm, FeedForward_Dropout])
        _Norm = LayerNormalization(name="Encoder-{}-FeedForward-Norm".format(layer_index))(FeedForward_Add)

    inputs = [Input_Token, Input_Segment]
    outputs = [_Norm]
    model = Model(inputs, outputs)
    #     adam = keras.optimizers.Adam(decay=0.2)
    if metrics is None:
        metrics = ['accuracy']
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=metrics)  # categorical_crossentropy, binary_crossentropy
    model.summary()
    return model


keras.backend.clear_session()
build_bert_model(output_layer_num=4, metrics=None)

from keras.engine.input_layer import InputLayer
from keras_bert.layers.embedding import TokenEmbedding
from keras.layers.embeddings import Embedding
from keras.layers.merge import Add
from keras_pos_embd.pos_embd import PositionEmbedding
from keras.layers.core import Dropout
from keras_layer_normalization.layer_normalization import LayerNormalization
from keras_multi_head.multi_head_attention import MultiHeadAttention
from keras_position_wise_feed_forward.feed_forward import FeedForward


def build_bert_model2(output_layer_num=4, metrics=None, output_dim=312):
    Input_Token = Input(shape=(None,), name="Input-Token")
    Input_Segment = Input(shape=(None,), name="Input-Segment")

    Embedding_Token = TokenEmbedding(input_dim=737, output_dim=output_dim, name="Embedding-Token")(Input_Token)
    Embedding_Segment = Embedding(input_dim=2, output_dim=output_dim, name="Embedding-Segment")(Input_Segment)

    Embedding_Token_Segment = Add(name="Embedding-Token-Segment")([Embedding_Token[0], Embedding_Segment])
    Embedding_Position = PositionEmbedding(input_dim=128, output_dim=output_dim, mode='add',
                                           embeddings_regularizer=None,
                                           embeddings_constraint=None,
                                           mask_zero=False,
                                           name="Embedding-Position")(Embedding_Token_Segment)
    Embedding_Dropout = Dropout(rate=0.1, name="Embedding-Dropout")(Embedding_Position)
    _Norm = LayerNormalization(name="Embedding-Norm")(Embedding_Dropout)
    for layer_index in range(1, output_layer_num + 1):
        MultiHeadSelfAttention = MultiHeadAttention(head_num=4,
                                                    name="Encoder-{}-MultiHeadSelfAttention".format(layer_index))(_Norm)
        MultiHeadSelfAttention_Dropout = Dropout(rate=0.1,
                                                 name="Encoder-{}-MultiHeadSelfAttention-Dropout".format(layer_index))(
            MultiHeadSelfAttention)
        MultiHeadSelfAttention_Add = Add(name="Encoder-{}-MultiHeadSelfAttention-Add".format(layer_index))(
            [_Norm, MultiHeadSelfAttention_Dropout])
        MultiHeadSelfAttention_Norm = LayerNormalization(
            name="Encoder-{}-MultiHeadSelfAttention-Norm".format(layer_index))(MultiHeadSelfAttention_Add)
        feedForward = FeedForward(units=64, name="Encoder-{}-FeedForward".format(layer_index))(
            MultiHeadSelfAttention_Norm)
        FeedForward_Dropout = Dropout(rate=0.1, name="Encoder-{}-FeedForward-Dropout".format(layer_index))(feedForward)
        FeedForward_Add = Add(name="Encoder-{}-FeedForward-Add".format(layer_index))(
            [MultiHeadSelfAttention_Norm, FeedForward_Dropout])
        _Norm = LayerNormalization(name="Encoder-{}-FeedForward-Norm".format(layer_index))(FeedForward_Add)
    lamb = keras.layers.Lambda(lambda x: K.mean(x, axis=1), name="lambda")(_Norm)
    dense1 = Dense(32, kernel_initializer="uniform", activation='relu')(lamb)
    dense1 = Dropout(0.2)(dense1)
    output = Dense(1, kernel_initializer="uniform", activation='sigmoid', name="output")(dense1)  # # softmax, sigmoid

    inputs = [Input_Token, Input_Segment]
    outputs = [output]
    model = Model(inputs, outputs)
    #     adam = keras.optimizers.Adam(decay=0.2)
    if metrics is None:
        metrics = ['accuracy']
    model.compile(loss='binary_crossentropy', optimizer='adam',
                  metrics=metrics)  # categorical_crossentropy, binary_crossentropy
    model.summary()
    return model


keras.backend.clear_session()
my_model = build_bert_model2(output_layer_num=4, metrics=None, output_dim=64)

# 从头构建RoBERTa-tiny-clue模型结构, 最终训练的时候，可能出现异常；具体表现：训练开始没多久后loss不降、accuracy不增，针对不同输入值，输出结果是一样的；
# 排除了损失函数、激活函数、正则化、归一化、数据集、padding等原因后，调整训练策略可解决：
# 即初始化后，先锁定靠近输入层的参数，训练模型至收敛；
# 待收敛后，解冻全部冻结层，降低学习率继续训练至收敛即可；
# 具体详情见：Sentence-RoBERT语义相似模型训练.py

def main():
    pass


if __name__ == '__main__':
    main()
