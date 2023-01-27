#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# 来源： https://github.com/bojone/bert4keras/tree/master/examples
# 句子对分类任务，LCQMC数据集
# val_acc: 0.887071, test_acc: 0.870320

import os

os.environ['TF_KERAS'] = '1'
USERNAME = os.getenv("USERNAME")
import numpy as np
import random
from bert4keras.backend import keras, set_gelu, K
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open
from keras.layers import Dropout, Dense
import tensorflow as tf
set_gelu('tanh')  # 切换gelu版本

maxlen = 128
batch_size = 32
# config_path = '/root/kg/bert/chinese_wwm_L-12_H-768_A-12/bert_config.json'
# checkpoint_path = '/root/kg/bert/chinese_wwm_L-12_H-768_A-12/bert_model.ckpt'
# dict_path = '/root/kg/bert/chinese_wwm_L-12_H-768_A-12/vocab.txt'


# Robert配置
# https://github.com/ZhuiyiTechnology/pretrained-models
# https://open.zhuiyi.ai/releases/nlp/models/zhuiyi/chinese_roberta_L-4_H-312_A-12.zip
config_path = rf'D:/Users/{USERNAME}/data/chinese_roberta_L-4_H-312_A-12/bert_config.json'
checkpoint_path = rf'D:/Users/{USERNAME}/data/chinese_roberta_L-4_H-312_A-12/bert_model.ckpt'
dict_path = rf'D:/Users/{USERNAME}/data/chinese_roberta_L-4_H-312_A-12/vocab.txt'


def load_data(filename):
    """加载数据
    单条格式：(文本1, 文本2, 标签id)
    """
    D = []
    with open(filename, encoding='utf-8') as f:
        for l in f:
            text1, text2, label = l.strip().split('\t')
            if text1 == 'text_a':
                continue
            D.append((text1, text2, int(label)))
    return D


# 加载数据集
# 数据来源：https://github.com/xiaohai-AI/lcqmc_data
train_data = load_data(rf'D:/Users/{USERNAME}/github_project/lcqmc_data/train.txt')
valid_data = load_data(rf'D:/Users/{USERNAME}/github_project/lcqmc_data/dev.txt')
test_data = load_data(rf'D:/Users/{USERNAME}/github_project/lcqmc_data/test.txt')

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)



def data_generator(train_data, batch_size=32, shuffle=True):
    while True:
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        if shuffle:
            random.shuffle(train_data)

        for text1, text2, label in train_data:
            token_ids, segment_ids = tokenizer.encode(
                text1, text2, maxlen=maxlen
            )
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([label])
            if len(batch_token_ids) == batch_size:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []

# 加载预训练模型
bert = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    with_pool=False,
    return_keras_model=False,
)

output = keras.layers.Lambda(lambda x: K.mean(x, axis=1) )(bert.model.outputs[-1])
output = Dropout(rate=0.1)(output)
output = Dense(
    units=2, activation='softmax', kernel_initializer=bert.initializer
)(output)

model = keras.models.Model(bert.model.input, output)
model.summary()

model.compile(
    loss='sparse_categorical_crossentropy',  # 其中 sparse 的含义是，真实的标签值 y_true 可以直接传入 int 类型的标签类别，即sparse时 y 不需要one-hot，而 categorical_crossentropy 需要。
    optimizer=Adam(2e-5),  # 用足够小的学习率
    # optimizer=PiecewiseLinearLearningRate(Adam(5e-5), {10000: 1, 30000: 0.1}),
    metrics=['accuracy'],
)

def evaluate(data):
    total, right = 0., 0.
    for x_true, y_true in data:
        y_pred = model.predict(x_true).argmax(axis=1)
        y_true = y_true[:, 0]
        total += len(y_true)
        right += (y_true == y_pred).sum()
    return right / total




def main():
    historys = model.fit(data_generator(train_data, batch_size=batch_size, shuffle=True),
                         steps_per_epoch=len(train_data) // batch_size,
                         # 在声明一个 epoch 完成并开始下一个 epoch 之前从 generator 产生的总步数（批次样本）。 它通常应该等于你的数据集的样本数量除以批量大小。len(train_data)/batch_size
                         validation_data=data_generator(valid_data, batch_size=batch_size, shuffle=True),
                         validation_steps=len(valid_data) // batch_size,
                         # 仅当 validation_data 是一个生成器时才可用。 在停止前 generator 生成的总步数（样本批数）。len(dev_data)/batch_size
                         epochs=2,
                         verbose=1,
                         shuffle=True,
                         )
    print(historys.history)



if __name__ == '__main__':
    main()
