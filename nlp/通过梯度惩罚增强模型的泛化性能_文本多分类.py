#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 通过梯度惩罚增强模型的泛化性能
# 比CLUE榜单公开的同数据集上的BERT base的成绩高2%
# 数据集：IFLYTEK' 长文本分类 (https://github.com/CLUEbenchmark/CLUE)
# 来源：https://kexue.fm/archives/7234， https://github.com/bojone/bert4keras/tree/master/examples


import os
os.environ['TF_KERAS'] = '1'
USERNAME = os.getenv("USERNAME")
import random
import json, copy
import numpy as np
import tensorflow.keras.backend as K
from tensorflow import keras
from bert4keras.backend import search_layer
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from tensorflow.keras.layers import Lambda, Dense
from keras.engine import data_adapter
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.losses import sparse_categorical_crossentropy, categorical_crossentropy
from tensorflow.python.ops import math_ops

num_classes = 119
maxlen = 128
batch_size = 32

# # BERT base
# config_path = rf'D:/Users/{USERNAME}/data/chinese_L-12_H-768_A-12/bert_config.json'
# checkpoint_path = rf'D:/Users/{USERNAME}/data/chinese_L-12_H-768_A-12/bert_model.ckpt'
# dict_path = rf'D:/Users/{USERNAME}/data/chinese_L-12_H-768_A-12/vocab.txt'

# Robert配置
# https://github.com/ZhuiyiTechnology/pretrained-models
# https://open.zhuiyi.ai/releases/nlp/models/zhuiyi/chinese_roberta_L-4_H-312_A-12.zip
config_path = rf'D:/Users/{USERNAME}/data/chinese_roberta_L-4_H-312_A-12/bert_config.json'
checkpoint_path = rf'D:/Users/{USERNAME}/data/chinese_roberta_L-4_H-312_A-12/bert_model.ckpt'
dict_path = rf'D:/Users/{USERNAME}/data/chinese_roberta_L-4_H-312_A-12/vocab.txt'

def load_data(filename):
    """加载数据
    单条格式：(文本, 标签id)
    """
    D = []
    with open(filename, encoding='utf-8') as f:
        for i, l in enumerate(f):
            l = json.loads(l)
            text, label = l['sentence'], l['label']
            D.append((text, int(label)))
    return D


# 加载数据集
train_data = load_data(
    rf'D:/Users/{USERNAME}/data/iflytek/train.json'
)
valid_data = load_data(
    rf'D:/Users/{USERNAME}/data/iflytek/dev.json'
)

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

def data_generator2(train_data, batch_size, shuffle=True):
    while True:
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        if shuffle:
            random.shuffle(train_data)
        for text, label in train_data:
            token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([label])
            if len(batch_token_ids) == batch_size:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield {'Input-Token': batch_token_ids, 'Input-Segment': batch_segment_ids}, batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []

def data_generator3(train_data, batch_size=None, shuffle=True):

    batch_token_ids, batch_segment_ids, batch_labels = [], [], []
    if shuffle:
        random.shuffle(train_data)
    for text, label in train_data:
        token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
        batch_token_ids.append(token_ids)
        batch_segment_ids.append(segment_ids)
        batch_labels.append([label])
    batch_token_ids = sequence_padding(batch_token_ids)
    batch_segment_ids = sequence_padding(batch_segment_ids)
    batch_labels = sequence_padding(batch_labels)
    return {'Input-Token': batch_token_ids, 'Input-Segment': batch_segment_ids}, batch_labels

# train_generator = data_generator2(train_data, batch_size)
# valid_generator = data_generator2(valid_data, batch_size)

# 加载预训练模型
bert = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    return_keras_model=False,
)

output = Lambda(lambda x: x[:, 0])(bert.model.output)
output = Dense(
    units=num_classes,
    activation='softmax',
    kernel_initializer=bert.initializer
)(output)

def sparse_categorical_crossentropy(y_true, y_pred):
    """自定义稀疏交叉熵
    这主要是因为keras自带的sparse_categorical_crossentropy不支持求二阶梯度。
    """
    y_true = K.reshape(y_true, K.shape(y_pred)[:-1])
    y_true = K.cast(y_true, 'int32')
    y_true = K.one_hot(y_true, K.shape(y_pred)[-1])
    return K.categorical_crossentropy(y_true, y_pred)

model = keras.models.Model(bert.model.input, output)
model.summary()

def loss_with_gradient_penalty(y_true, y_pred, epsilon=1):
    """带梯度惩罚的loss
    """
    loss = K.mean(sparse_categorical_crossentropy(y_true, y_pred))
    embeddings = model.get_layer('Embedding-Token').embeddings
    gp = K.sum(K.gradients(loss, [embeddings])[0].values**2)
    return loss + 0.5 * epsilon * gp

model.compile(
    loss=loss_with_gradient_penalty,
    optimizer=Adam(2e-5),
    metrics=['sparse_categorical_accuracy'],
)


def adversarial_training(model, embedding_name, epsilon=1):
    """给模型添加对抗训练
    其中model是需要添加对抗训练的keras模型，embedding_name
    则是model里边Embedding层的名字。要在模型compile之后使用。
    """
    if model.train_function is None:  # 如果还没有训练函数
        model._make_train_function()  # 手动make
    old_train_function = model.train_function  # 备份旧的训练函数

    # 查找Embedding层
    for output in model.outputs:
        embedding_layer = search_layer(output, embedding_name)
        if embedding_layer is not None:
            break
    if embedding_layer is None:
        raise Exception('Embedding layer not found')

    # 求Embedding梯度
    embeddings = embedding_layer.embeddings  # Embedding矩阵
    gradients = K.gradients(model.total_loss, [embeddings])  # Embedding梯度
    gradients = K.zeros_like(embeddings) + gradients[0]  # 转为dense tensor

    # 封装为函数
    inputs = (
        model._feed_inputs + model._feed_targets + model._feed_sample_weights
    )  # 所有输入层
    embedding_gradients = K.function(
        inputs=inputs,
        outputs=[gradients],
        name='embedding_gradients',
    )  # 封装为函数

    def train_function(inputs):  # 重新定义训练函数
        grads = embedding_gradients(inputs)[0]  # Embedding梯度
        delta = epsilon * grads / (np.sqrt((grads**2).sum()) + 1e-8)  # 计算扰动
        K.set_value(embeddings, K.eval(embeddings) + delta)  # 注入扰动
        outputs = old_train_function(inputs)  # 梯度下降
        K.set_value(embeddings, K.eval(embeddings) - delta)  # 删除扰动
        return outputs

    model.train_function = train_function  # 覆盖原训练函数


# 写好函数后，启用对抗训练只需要一行代码
# adversarial_training(model, 'Embedding-Token', 0.5)

def predict_to_file(in_file, out_file):
    """输出预测结果到文件
    结果文件可以提交到 https://www.cluebenchmarks.com 评测。
    """
    fw = open(out_file, 'w', encoding='utf-8')
    with open(in_file, encoding='utf-8') as fr:
        for l in tqdm(fr):
            l = json.loads(l)
            text = l['sentence']
            token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
            label = model.predict([[token_ids], [segment_ids]])[0].argmax()
            l = json.dumps({'id': str(l['id']), 'label': str(label)})
            fw.write(l + '\n')
    fw.close()


def main():

    historys = model.fit(data_generator2(train_data, batch_size),
                                   steps_per_epoch=len(train_data)//batch_size,
                                   # 在声明一个 epoch 完成并开始下一个 epoch 之前从 generator 产生的总步数（批次样本）。 它通常应该等于你的数据集的样本数量除以批量大小。len(train_data)/batch_size
                                   validation_data=data_generator2(valid_data, batch_size),
                                   validation_steps=len(valid_data)//batch_size,
                                   # 仅当 validation_data 是一个生成器时才可用。 在停止前 generator 生成的总步数（样本批数）。len(dev_data)/batch_size
                                   epochs=50,
                                   verbose=1,
                                   shuffle=True,
                                   )

    train_x, train_y = data_generator3(train_data[:100], batch_size)
    historys = model.fit(x=train_x, y=train_y,
                         validation_data=data_generator3(valid_data[:40], batch_size),
                         batch_size=batch_size,
                         # 仅当 validation_data 是一个生成器时才可用。 在停止前 generator 生成的总步数（样本批数）。len(dev_data)/batch_size
                         epochs=1,
                         verbose=1,
                         shuffle=True
                         )
    print(historys.history)

    # model.load_weights('best_model.weights')
    predict_to_file(rf'D:/Users/{USERNAME}/data/iflytek/test.json', 'iflytek_predict.json')

if __name__ == '__main__':
    main()
