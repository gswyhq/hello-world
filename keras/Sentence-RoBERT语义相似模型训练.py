#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import json
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
from keras.optimizers.optimizer_v2.adam import Adam
from keras.initializers.initializers_v2 import RandomUniform
import matplotlib.pyplot as plt
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei'] #指定默认字体 
mpl.rcParams['axes.unicode_minus'] = False #解决保存图像是负号'-'显示为方块的问题
from keras.engine.input_layer import InputLayer
from keras_bert.layers.embedding import TokenEmbedding
# from keras.layers.embeddings import Embedding  # keras version==2.6.0
# from keras.layers.merge import Add
from keras.layers.core.embedding import Embedding  # keras version == 2.9.0
from keras.layers.merging.add import Add
from keras_pos_embd.pos_embd import PositionEmbedding
from keras.layers.core import Dropout
from keras_layer_normalization.layer_normalization import LayerNormalization
from keras_multi_head.multi_head_attention import MultiHeadAttention
from keras_position_wise_feed_forward.feed_forward import FeedForward
from keras.layers import Dense, Dropout, Lambda, Input, Concatenate
from keras.losses import MeanSquaredError, cosine_similarity
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix, average_precision_score, precision_recall_curve

# 预训练模型来源：https://github.com/CLUEbenchmark/CLUEPretrainedModels.git
config_path = r'D:\Users\{}\data\RoBERTa-tiny-clue\bert_config.json'.format(os.getenv("USERNAME"))
checkpoint_path = r'D:\Users\{}\data\RoBERTa-tiny-clue\bert_model.ckpt'.format(os.getenv("USERNAME"))
dict_path = r'D:\Users\{}\data\RoBERTa-tiny-clue\vocab.txt'.format(os.getenv("USERNAME"))


token_dict = {}
with open(dict_path, 'r', encoding='utf-8') as f:
    for line in f.readlines():
        token = line.strip()
        token_dict[token] = len(token_dict)

tokenizer = Tokenizer(token_dict, pad_index=0)
cut_words = tokenizer.tokenize(u'今天天气不错')
print(cut_words)

VOCAB_SIZE = 8021

def defined_shared_layers(output_layer_num=4, output_dim=312, forward_units=1248, input_emb_dim=512, head_num=12):
    """共享图层实例化"""
    shared_layers_dict = {}
    Embedding_Token = TokenEmbedding(input_dim=VOCAB_SIZE, output_dim=output_dim, name="Embedding-Token",
                                     embeddings_initializer=tf.keras.initializers.RandomUniform(minval=-0.05,
                                                                                                maxval=0.05),
                                     embeddings_regularizer=None,
                                     embeddings_constraint=None,
                                     mask_zero=True,
                                     )
    Embedding_Segment = Embedding(input_dim=2, output_dim=output_dim, name="Embedding-Segment",
                                  embeddings_initializer=tf.keras.initializers.RandomUniform(minval=-0.05,
                                                                                             maxval=0.05),
                                  embeddings_regularizer=None,
                                  embeddings_constraint=None,
                                  mask_zero=False,
                                  )

    Embedding_Token_Segment = Add(name="Embedding-Token-Segment")
    Embedding_Position = PositionEmbedding(input_dim=input_emb_dim, output_dim=output_dim, mode= 'add',
                                           embeddings_initializer = tf.keras.initializers.RandomUniform(minval=-0.05, maxval=0.05),
                                           embeddings_regularizer= None,
                                           embeddings_constraint= None,
                                           mask_zero= False,
                                           name="Embedding-Position")
    Embedding_Dropout = Dropout(rate=0.1, name="Embedding-Dropout")
    _Norm = LayerNormalization(name="Embedding-Norm",
                               center= True, scale= True, epsilon= 1e-14, gamma_initializer=tf.keras.initializers.Ones(),
                               beta_initializer= tf.keras.initializers.Zeros() ,
                               gamma_regularizer= None, beta_regularizer= None, gamma_constraint= None, beta_constraint= None
    )
    for shared_layer in [Embedding_Token, Embedding_Segment, Embedding_Token_Segment, Embedding_Position, Embedding_Dropout, _Norm]:
        shared_layers_dict[shared_layer.name] = shared_layer
    for layer_index in range(1, output_layer_num+1):
        MultiHeadSelfAttention = MultiHeadAttention(head_num=head_num, name="Encoder-{}-MultiHeadSelfAttention".format(layer_index),
                                                    activation='linear', use_bias= True, kernel_initializer= tf.keras.initializers.GlorotNormal(),
                                                    bias_initializer=tf.keras.initializers.Zeros() ,
                                                    kernel_regularizer= None, bias_regularizer= None, kernel_constraint= None, bias_constraint= None, history_only= False)
        MultiHeadSelfAttention_Dropout = Dropout(rate=0.1, name="Encoder-{}-MultiHeadSelfAttention-Dropout".format(layer_index))
        MultiHeadSelfAttention_Add = Add(name="Encoder-{}-MultiHeadSelfAttention-Add".format(layer_index))
        MultiHeadSelfAttention_Norm = LayerNormalization(name="Encoder-{}-MultiHeadSelfAttention-Norm".format(layer_index),
                                                         center=True, scale=True, epsilon=1e-14,
                                                         gamma_initializer=tf.keras.initializers.Ones(),
                                                         beta_initializer=tf.keras.initializers.Zeros(),
                                                         gamma_regularizer=None, beta_regularizer=None,
                                                         gamma_constraint=None, beta_constraint=None
                                                         )
        feedForward = FeedForward(units=forward_units, name="Encoder-{}-FeedForward".format(layer_index),
                                  activation='gelu', use_bias= True, kernel_initializer= tf.keras.initializers.GlorotNormal(),
                                  bias_initializer= tf.keras.initializers.Zeros(),
                                  kernel_regularizer= None, bias_regularizer= None, kernel_constraint= None, bias_constraint= None, dropout_rate= 0.0)
        FeedForward_Dropout = Dropout(rate=0.1, name="Encoder-{}-FeedForward-Dropout".format(layer_index))
        FeedForward_Add = Add(name="Encoder-{}-FeedForward-Add".format(layer_index))
        _Norm = LayerNormalization(name="Encoder-{}-FeedForward-Norm".format(layer_index),
                                   center=True, scale=True, epsilon=1e-14,
                                   # gamma_initializer = tf.keras.initializers.Constant(1.2),
                                   gamma_initializer=tf.keras.initializers.Ones(),
                                   beta_initializer=tf.keras.initializers.Zeros(),
                                   gamma_regularizer=None, beta_regularizer=None,
                                   gamma_constraint=None, beta_constraint=None
                                   )
        for shared_layer in [MultiHeadSelfAttention, MultiHeadSelfAttention_Dropout, MultiHeadSelfAttention_Add, MultiHeadSelfAttention_Norm,
                              feedForward, FeedForward_Dropout, FeedForward_Add, _Norm]:
            shared_layers_dict[shared_layer.name] = shared_layer
    lamb = keras.layers.Lambda(lambda x: K.mean(x, axis=1), name="lambda" )
    # lamb = keras.layers.GlobalAveragePooling1D(name="lambda")
    drop = Dropout(0.2, name="dropout")
    dense1 = Dense(16, name="dense", kernel_initializer="uniform", activation='relu')
    for shared_layer in [lamb, drop, dense1]:
        shared_layers_dict[shared_layer.name] = shared_layer
    return shared_layers_dict
SHARED_LAYERS_DICT = defined_shared_layers(output_layer_num=4, output_dim=64, forward_units=64, input_emb_dim=128, head_num=4)  # 共享层实例化，调整模型参数
# SHARED_LAYERS_DICT = defined_shared_layers(output_layer_num=4, output_dim=312, forward_units=1248, input_emb_dim=512, head_num=12)

def build_shared_network(Input_Token, Input_Segment, output_layer_num=4):
    """构建权重共享网络"""
    Embedding_Token = SHARED_LAYERS_DICT["Embedding-Token"](Input_Token)
    Embedding_Segment = SHARED_LAYERS_DICT["Embedding-Segment"](Input_Segment)

    Embedding_Token_Segment = SHARED_LAYERS_DICT["Embedding-Token-Segment"]([Embedding_Token[0], Embedding_Segment])
    Embedding_Position = SHARED_LAYERS_DICT["Embedding-Position"](Embedding_Token_Segment)
    Embedding_Dropout = SHARED_LAYERS_DICT["Embedding-Dropout"](Embedding_Position)
    _Norm = SHARED_LAYERS_DICT["Embedding-Norm"](Embedding_Dropout)

    for layer_index in range(1, output_layer_num + 1):
        MultiHeadSelfAttention = SHARED_LAYERS_DICT["Encoder-{}-MultiHeadSelfAttention".format(layer_index)](_Norm)
        MultiHeadSelfAttention_Dropout = SHARED_LAYERS_DICT[
            "Encoder-{}-MultiHeadSelfAttention-Dropout".format(layer_index)](MultiHeadSelfAttention)
        MultiHeadSelfAttention_Add = SHARED_LAYERS_DICT["Encoder-{}-MultiHeadSelfAttention-Add".format(layer_index)](
            [_Norm, MultiHeadSelfAttention_Dropout])
        MultiHeadSelfAttention_Norm = SHARED_LAYERS_DICT["Encoder-{}-MultiHeadSelfAttention-Norm".format(layer_index)](
            MultiHeadSelfAttention_Add)
        feedForward = SHARED_LAYERS_DICT["Encoder-{}-FeedForward".format(layer_index)](MultiHeadSelfAttention_Norm)
        FeedForward_Dropout = SHARED_LAYERS_DICT["Encoder-{}-FeedForward-Dropout".format(layer_index)](feedForward)
        FeedForward_Add = SHARED_LAYERS_DICT["Encoder-{}-FeedForward-Add".format(layer_index)](
            [MultiHeadSelfAttention_Norm, FeedForward_Dropout])
        _Norm = SHARED_LAYERS_DICT["Encoder-{}-FeedForward-Norm".format(layer_index)](FeedForward_Add)

    lamb = SHARED_LAYERS_DICT["lambda"](_Norm)
    drop = SHARED_LAYERS_DICT["dropout"](lamb)
    dense1 = SHARED_LAYERS_DICT["dense"](drop)
    return dense1

def build_shared_network2(output_layer_num=4):
    """构建权重共享网络"""
    Input_Token = Input(shape=(None,), name="Input-Token")
    Input_Segment = Input(shape=(None,), name="Input-Segment")
    Embedding_Token = SHARED_LAYERS_DICT["Embedding-Token"](Input_Token)
    Embedding_Segment = SHARED_LAYERS_DICT["Embedding-Segment"](Input_Segment)

    Embedding_Token_Segment = SHARED_LAYERS_DICT["Embedding-Token-Segment"]([Embedding_Token[0], Embedding_Segment])
    Embedding_Position = SHARED_LAYERS_DICT["Embedding-Position"](Embedding_Token_Segment)
    Embedding_Dropout = SHARED_LAYERS_DICT["Embedding-Dropout"](Embedding_Position)
    _Norm = SHARED_LAYERS_DICT["Embedding-Norm"](Embedding_Dropout)

    for layer_index in range(1, output_layer_num + 1):
        MultiHeadSelfAttention = SHARED_LAYERS_DICT["Encoder-{}-MultiHeadSelfAttention".format(layer_index)](_Norm)
        MultiHeadSelfAttention_Dropout = SHARED_LAYERS_DICT[
            "Encoder-{}-MultiHeadSelfAttention-Dropout".format(layer_index)](MultiHeadSelfAttention)
        MultiHeadSelfAttention_Add = SHARED_LAYERS_DICT["Encoder-{}-MultiHeadSelfAttention-Add".format(layer_index)](
            [_Norm, MultiHeadSelfAttention_Dropout])
        MultiHeadSelfAttention_Norm = SHARED_LAYERS_DICT["Encoder-{}-MultiHeadSelfAttention-Norm".format(layer_index)](
            MultiHeadSelfAttention_Add)
        feedForward = SHARED_LAYERS_DICT["Encoder-{}-FeedForward".format(layer_index)](MultiHeadSelfAttention_Norm)
        FeedForward_Dropout = SHARED_LAYERS_DICT["Encoder-{}-FeedForward-Dropout".format(layer_index)](feedForward)
        FeedForward_Add = SHARED_LAYERS_DICT["Encoder-{}-FeedForward-Add".format(layer_index)](
            [MultiHeadSelfAttention_Norm, FeedForward_Dropout])
        _Norm = SHARED_LAYERS_DICT["Encoder-{}-FeedForward-Norm".format(layer_index)](FeedForward_Add)

    lamb = SHARED_LAYERS_DICT["lambda"](_Norm)
    drop = SHARED_LAYERS_DICT["dropout"](lamb)
    dense1 = SHARED_LAYERS_DICT["dense"](drop)
    model = Model([Input_Token, Input_Segment], [dense1])  # 共享参数部分，需要定义输入输出，否则训练的时候可能出现问题
    return model

def build_network2(output_layer_num=4, output_dim=64, Input_Token=None, Input_Segment=None):
    # Input_Token = Input(shape=(None,), name="Input-Token")
    # Input_Segment = Input(shape=(None,), name="Input-Segment")

    Embedding_Token = TokenEmbedding(input_dim=VOCAB_SIZE, output_dim=output_dim)(Input_Token)
    Embedding_Segment = Embedding(input_dim=2, output_dim=output_dim, )(Input_Segment)

    Embedding_Token_Segment = Add()([Embedding_Token[0], Embedding_Segment])
    Embedding_Position = PositionEmbedding(input_dim=128, output_dim=output_dim, mode='add',
                                           embeddings_regularizer=None,
                                           embeddings_constraint=None,
                                           mask_zero=False,
                                           )(Embedding_Token_Segment)
    Embedding_Dropout = Dropout(rate=0.1, )(Embedding_Position)
    _Norm = LayerNormalization()(Embedding_Dropout)
    for layer_index in range(1, output_layer_num + 1):
        MultiHeadSelfAttention = MultiHeadAttention(head_num=4,)(_Norm)
        MultiHeadSelfAttention_Dropout = Dropout(rate=0.1,)(
            MultiHeadSelfAttention)
        MultiHeadSelfAttention_Add = Add()(
            [_Norm, MultiHeadSelfAttention_Dropout])
        MultiHeadSelfAttention_Norm = LayerNormalization()(MultiHeadSelfAttention_Add)
        feedForward = FeedForward(units=64, )(
            MultiHeadSelfAttention_Norm)
        FeedForward_Dropout = Dropout(rate=0.1, )(feedForward)
        FeedForward_Add = Add()(
            [MultiHeadSelfAttention_Norm, FeedForward_Dropout])
        _Norm = LayerNormalization()(FeedForward_Add)

    lamb = keras.layers.Lambda(lambda x: K.mean(x, axis=1), )(_Norm)
    drop = Dropout(0.2, )(lamb)
    dense1 = Dense(16, kernel_initializer="uniform", activation='relu')(drop)
    return dense1

def build_bert_model3(metrics=None):
    """从头构建网络，定义输入输出"""
    Input_Token_a = Input(shape=(None,), name="Input-Token-a")
    Input_Segment_a = Input(shape=(None,), name="Input-Segment-a")
    Input_Token_b = Input(shape=(None,), name="Input-Token-b")
    Input_Segment_b = Input(shape=(None,), name="Input-Segment-b")

    # 方法1：共享权重，仅仅使用一次Model，构建出的模型，会输出子模型的详情
    dense_a = build_shared_network(Input_Token_a, Input_Segment_a)
    dense_b = build_shared_network(Input_Token_b, Input_Segment_b)

    # 方法2：不共享权重
    # dense_a = build_network2(output_layer_num=4, output_dim=64, Input_Token=Input_Token_a, Input_Segment=Input_Segment_a)
    # dense_b = build_network2(output_layer_num=4, output_dim=64, Input_Token=Input_Token_b, Input_Segment=Input_Segment_b)

    # 方法3：共享权重，两次使用Model，构建出的模型，会掩盖掉子模型的详情
    # share_model = build_shared_network2()
    # dense_a = share_model({"Input-Token": Input_Token_a,
    #                          "Input-Segment": Input_Segment_a,
    #                          })
    # dense_b = share_model({"Input-Token": Input_Token_b,
    #                          "Input-Segment": Input_Segment_b,
    #                          })

    concat_layer = Concatenate()([dense_a, dense_b, K.abs(dense_a - dense_b)])

    output = Dense(2, kernel_initializer="uniform", activation='softmax', name="output")(concat_layer)  # softmax, sigmoid
    inputs = [Input_Token_a, Input_Segment_a, Input_Token_b, Input_Segment_b]
    outputs = [output]
    model = Model(inputs, outputs)
    #     adam = keras.optimizers.Adam(decay=0.2)
    if metrics is None:
        metrics = ['accuracy']

    model.compile(loss='categorical_crossentropy',  # categorical_crossentropy: acc:0.5, binary_crossentropy
                  optimizer='adam',
                  metrics=metrics)  # categorical_crossentropy, binary_crossentropy
    model.summary()
    return model


keras.backend.clear_session()
# shared_model = build_bert_model3(metrics=None)

task_name = ''
def load_data(filename):
    """加载数据（带标签）
    单条格式：(文本1, 文本2, 标签)
    """
    D = []
    df = pd.read_csv(filename, sep='\t', encoding='utf-8')
    for text_a, text_b, label in df[ ['text_a', 'text_b', 'label']].values:
        D.append((text_a, text_b, float(label)))
    return D

# 数据来源：https://github.com/xiaohai-AI/lcqmc_data
USERNAME = os.getenv("USERNAME")
data_path = rf'D:\Users\{USERNAME}\github_project\lcqmc_data'
datasets = {
    '%s-%s' % (task_name, f):
    load_data('%s/%s.txt' % (data_path, f))
    for f in ['train', 'dev', 'test']
}


train_data = datasets['-train']
dev_data = datasets['-dev']
test_data = datasets['-test']
# train_data = train_data[:100]
# dev_data = dev_data[:10]
# test_data = test_data[:100]

train_token_ids_a = [tokenizer.encode(t[0], second=None, max_len=46)[0] for t in train_data]
train_token_ids_b = [tokenizer.encode(t[1], second=None, max_len=46)[0] for t in train_data]
dev_token_ids_a = [tokenizer.encode(t[0], second=None, max_len=46)[0] for t in dev_data]
dev_token_ids_b = [tokenizer.encode(t[1], second=None, max_len=46)[0] for t in dev_data]
test_token_ids_a = [tokenizer.encode(t[0], second=None, max_len=46)[0] for t in test_data]
test_token_ids_b = [tokenizer.encode(t[1], second=None, max_len=46)[0] for t in test_data]

train_token_ids_a = np.array(train_token_ids_a)
train_token_ids_b = np.array(train_token_ids_b)
dev_token_ids_a = np.array(dev_token_ids_a)
dev_token_ids_b = np.array(dev_token_ids_b)
test_token_ids_a = np.array(test_token_ids_a)
test_token_ids_b = np.array(test_token_ids_b)

train_x = {
    "Input-Token-a": train_token_ids_a,
    "Input-Segment-a": np.zeros_like(train_token_ids_a),
    "Input-Token-b": train_token_ids_b,
    "Input-Segment-b": np.zeros_like(train_token_ids_b),
          }
# train_y = np.reshape([t[-1] for t in train_similarity_data], (len(train_similarity_data), 1))
train_y = tf.keras.utils.to_categorical([int(t[-1]) for t in train_data], num_classes=2)

validation_x = {
    "Input-Token-a": dev_token_ids_a,
    "Input-Segment-a": np.zeros_like(dev_token_ids_a),
    "Input-Token-b": dev_token_ids_b,
    "Input-Segment-b": np.zeros_like(dev_token_ids_b),
          }

# validation_y = np.reshape([t[-1] for t in dev_similarity_data], (len(dev_similarity_data), 1))
validation_y = tf.keras.utils.to_categorical([int(t[-1]) for t in dev_data], num_classes=2)
print(train_y.shape, sum(train_y[:,1]), validation_y.shape, sum(validation_y[:,1]))

test_x = {
    "Input-Token-a": test_token_ids_a,
    "Input-Segment-a": np.zeros_like(test_token_ids_a),
    "Input-Token-b": test_token_ids_b,
    "Input-Segment-b": np.zeros_like(test_token_ids_b),
          }
test_y = tf.keras.utils.to_categorical([int(t[-1]) for t in test_data], num_classes=2)


def build_bert_model4(trainable=False):
    '''加载预训练模型获取特征向量，并构建模型'''
    keras.backend.clear_session()
    bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=None)
    for l in bert_model.layers:
        l.trainable = trainable
    output = bert_model.get_layer(
        'Encoder-%d-FeedForward-Norm' % 4
    ).output
    output = keras.layers.Lambda(lambda x: K.mean(x, axis=1))(output)
    # output = keras.layers.GlobalAveragePooling1D()(output)
    output = Dropout(0.2, )(output)
    output = Dense(16, kernel_initializer="uniform", activation='relu')(output)
    feature_model = Model(bert_model.inputs, output)

    Input_Token_a = Input(shape=(None,), name="Input-Token-a")
    Input_Segment_a = Input(shape=(None,), name="Input-Segment-a")
    Input_Token_b = Input(shape=(None,), name="Input-Token-b")
    Input_Segment_b = Input(shape=(None,), name="Input-Segment-b")

    dense_a = feature_model( {"Input-Token": Input_Token_a,
                               "Input-Segment": Input_Segment_a,
                              })
    dense_b = feature_model( {"Input-Token": Input_Token_b,
                               "Input-Segment": Input_Segment_b,
                              })

    concat_layer = Concatenate()([dense_a, dense_b, K.abs(dense_a - dense_b)])
    # concat_layer = Dense(16, activation='relu')(concat_layer)
    output = Dense(2, kernel_initializer="uniform", activation='softmax', name="output")(
        concat_layer)  # softmax, sigmoid
    inputs = [Input_Token_a, Input_Segment_a, Input_Token_b, Input_Segment_b]
    outputs = [output]
    model = Model(inputs, outputs)
    #     adam = keras.optimizers.Adam(decay=0.2)
    metrics = ['accuracy']

    model.compile(loss='categorical_crossentropy',  # categorical_crossentropy: acc:0.5, mse,
                  optimizer='adam',
                  metrics=metrics)  # categorical_crossentropy, binary_crossentropy
    model.summary()
    return model

def test_1():
    """RoBERTa-tiny-clue，先锁定预训练参数，训练收敛后，再微调，训练正常；"""
    # 第一步，锁定预训练层参数，训练模型至收敛
    model = build_bert_model4(trainable=False)
    historys = model.fit(x=train_x,
                         y=train_y,
                         validation_data=(validation_x, validation_y),
                         epochs=20,
                         batch_size=8,
                         verbose=1,
                         #         sample_weight=sample_weights,
                         shuffle=True,
                         # callbacks=[early_stopping, model_checkpoint, tb]
                         )

    print(historys.history)
    # Total params: 7,357,386
    # Trainable params: 5,106
    # Non-trainable params: 7,352,280
    # __________________________________________________________________________________________________
    #  2994/29846 [==>...........................] - ETA: 1:37:12 - loss: 0.6484 - accuracy: 0.6256
    # {"loss": [0.6175205707550049, 0.6089749336242676, 0.6057991981506348, 0.6051381826400757, 0.6041877865791321],
    # "accuracy": [0.6577611565589905, 0.665840208530426, 0.6693959832191467, 0.6697394251823425, 0.6707361936569214],
    # "val_loss": [0.7594574689865112, 0.7448214292526245, 0.8079662322998047, 0.7260235548019409, 0.7217984199523926],
    # "val_accuracy": [0.5473756194114685, 0.5437400341033936, 0.5343104004859924, 0.5649852156639099, 0.556350827217102]}
    # 训练完后，训练集准确率0.67，校验集上准确率(val_accuracy)有0.55；但调整阈值后，发现测试集上准确率也有0.67；
    result2 = model.predict(test_x)
    thr = 0.77
    ret_mat = confusion_matrix([t[-1] for t in test_data], [1 if t>=thr else 0 for t in result2[:,1]])
    ret_mat, (ret_mat[0,0]+ret_mat[1,1])/ret_mat.sum()
    # Out[119]:
    # (array([[4163, 2087],
    #         [2000, 4250]], dtype=int64),
    #  0.67304)
    # 保存初步训练的模型
    model.save(rf"D:\Users\{USERNAME}\Downloads\test\bert_model2.h5")

    # 加载初步训练的模型并进行微调
    bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=None)
    custom_objects = {layer.__class__.__name__:layer for layer in bert_model.layers}
    model2 = load_model(rf"D:\Users\{USERNAME}\Downloads\test\bert_model2.h5", custom_objects=custom_objects)
    # 将所有图层的参数设置为皆可训练
    model_2 = model2.get_layer("model_2")  # 因为Model
    for l in model_2.layers:
        l.trainable = True
    # 降低学习率，训练模型
    model2.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model2.summary()

    historys = model2.fit(x=train_x,
                         y=train_y,
                         validation_data=(validation_x, validation_y),
                         epochs=2,
                         batch_size=8,
                         verbose=1,
                         #         sample_weight=sample_weights,
                         shuffle=True,
                         # callbacks=[early_stopping, model_checkpoint, tb]
                         )

    print(historys.history)
    # ==================================================================================================
    # Total params: 7,357,386
    # Trainable params: 7,357,386
    # Non-trainable params: 0
    # 163/29846 loss: 0.5950 accuracy: 0.6776
    # 2208/29846 loss: 0.5618 accuracy: 0.7078
    #  4422/29846 [===>..........................] - ETA: 4:47:39 - loss: 0.5420 - accuracy: 0.7255
    # model2.save(rf"D:\Users\{USERNAME}\Downloads\test\bert_model3.h5")
    # {'loss': [0.4534868001937866, 0.3755546510219574], 'accuracy': [0.789743959903717, 0.8370454907417297],
    # 'val_loss': [0.6096696257591248, 0.55177903175354], 'val_accuracy': [0.7207452654838562, 0.7447171211242676]}
    # thr = 0.85
    # ret_mat = confusion_matrix([t[-1] for t in test_data], [1 if t >= thr else 0 for t in result2[:, 1]])
    # ret_mat, (ret_mat[0, 0] + ret_mat[1, 1]) / ret_mat.sum()
    # Out[17]:
    # (array([[5242, 1008],
    #         [1074, 5176]], dtype=int64),
    #  0.83344)
    # 微调后（未2轮训练），调整阈值，测试集上准确率有83%。
    # 效果较差，对比：https://paperswithcode.com/sota/chinese-sentence-pair-classification-on-lcqmc， Glyce + BERT， acc: 88.7%

def test_2():
    """
    RoBERTa-tiny-clue，不锁定预训练参数，直接从零训练，训练异常，表现为刚开始训练没多久，就所有输出为固定值；
    最后模型训练loss保持不变；所有测试数据输入后，输出都是固定值；
    """
    model = build_bert_model4(trainable=True)
    historys = model.fit(x=train_x,
                         y=train_y,
                         validation_data=(validation_x, validation_y),
                         epochs=20,
                         batch_size=8,
                         verbose=1,
                         #         sample_weight=sample_weights,
                         shuffle=True,
                         # callbacks=[early_stopping, model_checkpoint, tb]
                         )

    print(historys.history)
    # 29846/29846 [==============================] - 2407s 81ms/step - loss: 0.6613 - accuracy: 0.6130 - val_loss: 0.7058 - val_accuracy: 0.5006

def test_3():
    """# 自定义RoBERTa-tiny-clue网络结构，从零训练，训练异常同上；
    异常详情：训练开始后没多久（不等一轮训练完，约输入一两百个批次数据即出现），
    对应不同的模型输入，输出结果是一致的。导致模型loss保持不降，accuracy不增加
    针对该问题：采用，不同的损失函数，激活函数，padding，正则化策略均失效，也就是说这些均不是原因"""
    model = build_bert_model3(metrics=['accuracy'])
    historys = model.fit(x=train_x,
                         y=train_y,
                         validation_data=(validation_x, validation_y),
                         epochs=20,
                         batch_size=8,
                         verbose=1,
                         #         sample_weight=sample_weights,
                         shuffle=True,
                         # callbacks=[early_stopping, model_checkpoint, tb]
                         )

    print(historys.history)

    # 训练时候是存在问题的(预测的时候所有输入，其输出结果一致)
    # 29846/29846 [==============================] - 2407s 81ms/step - loss: 0.6613 - accuracy: 0.6130 - val_loss: 0.7058 - val_accuracy: 0.5006
    test_x3 = {'Input-Token': test_x['Input-Token-a'], 'Input-Segment':test_x['Input-Segment-a']}
    sub_model = Model([model.get_layer("Embedding-Token").input, model.get_layer("Embedding-Segment").input, ], model.get_layer("Encoder-2-MultiHeadSelfAttention").output)
    sub_model(test_x3)

def test_4():
    """
    针对训练异常（训练开始后没多久，模型就输出固定值，train loss不降，accuracy不增），排除模型结构、损失函数、激活函数、正则化、归一化、数据集等原因后
    采取策略是冻结较低层，至模型收敛；再解冻较低层，降低学习率训练；最后训练得以正常
    """
    model = build_bert_model3(metrics=['accuracy'])

    # 初始化后，先冻结较低层，训练模型
    for l in model.layers[:-7]:
        l.trainable = False

    model.compile(
        optimizer='adam',
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    historys = model.fit(x=train_x,
                         y=train_y,
                         validation_data=(validation_x, validation_y),
                         epochs=20,
                         batch_size=8,
                         verbose=1,
                         #         sample_weight=sample_weights,
                         shuffle=True,
                         # callbacks=[early_stopping, model_checkpoint, tb]
                         )
    print(historys.history)
    # 训练一轮后，在验证集上val_accuracy 才0.47，比随机预测(0.5)的结果都低；
    # 但无所谓，因为此时，模型输出结果是不同的；针对前面的异常情况（不管怎么训练模型输出都是一样的）已经破局了；
    # 29845/29846 [============================>.] - ETA: 0s - loss: 0.6618 - accuracy: 0.6131 - val_loss: 0.7580 - val_accuracy: 0.4747

    # 待训练收敛后，再解冻所有层（未尝试逐渐解冻一个或两个较高隐藏层），降低学习率训练；
    for l in model.layers:
        l.trainable = True
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    historys = model.fit(x=train_x,
                         y=train_y,
                         validation_data=(validation_x, validation_y),
                         epochs=1,
                         batch_size=8,
                         verbose=1,
                         #         sample_weight=sample_weights,
                         shuffle=True,
                         # callbacks=[early_stopping, model_checkpoint, tb]
                         )
    print(historys.history)
    # 29846/29846 [==============================] - 2112s 70ms/step - loss: 0.6042 - accuracy: 0.6792 - val_loss: 0.7793 - val_accuracy: 0.5383

    # 训练一轮后，在训练集上accuracy为0.6792，校验集上val_accuracy为：0.5383；
    # 针对测试集，调整阈值后，accuracy,也有0.66184
    # 该结果较在预训练模型参数上微调的效果差；
    # 所以若可能的话，该用预训练模型参数就用预训练模型参数；
    # 否则从零训练的话，模型能不能收敛、能不能好好训练就难说，更何况即使能好好训练了但最后的效果也差
    thr = 0.725
    ret_mat = confusion_matrix([t[-1] for t in test_data], [1 if t>=thr else 0 for t in result2[:,1]])
    ret_mat, (ret_mat[0,0]+ret_mat[1,1])/ret_mat.sum()
    # Out[37]:
    # (array([[4115, 2135],
    #         [2092, 4158]], dtype=int64),
    #  0.66184)

def test_5():
    '''获取中间层输出结果的方法'''
    test_x3 = {'Input-Token': test_x['Input-Token-a'], 'Input-Segment':test_x['Input-Segment-a']}
    sub_model = Model([model.get_layer("Embedding-Token").input, model.get_layer("Embedding-Segment").input, ], model.get_layer("Encoder-2-MultiHeadSelfAttention").output)
    sub_model(test_x3)

def main():
    # 总结起来
    # 实验1：先锁定RoBERTa-tiny-clue预训练参数，训练收敛后，再微调(解冻冻结层降低学习率)，训练正常；
    test_1()

    # 实验2：不锁定RoBERTa-tiny-clue预训练参数，仅仅使用模型结构，直接从零训练，训练异常，表现为刚开始训练没多久，就所有输出为固定值；
    test_2()

    # 实验3：自定义RoBERTa-tiny-clue模型结构，直接从零训练，训练异常，表现为刚开始训练没多久，就所有输出为固定值；
    test_3()

    # 实验4：自定义RoBERTa-tiny-clue模型结构，锁定与实验1同等层参数（该参数不是预训练模型参数，而是自定义模型后，初始化参数）；
    # 锁定参数后，直接从零训练，直至收敛；再解冻所有锁定层，降低学习率，训练直至收敛；
    # 最后训练正常，只不过效果可能没有实验1效果好；
    test_4()

if __name__ == '__main__':
    main()
