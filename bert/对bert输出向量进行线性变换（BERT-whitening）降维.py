#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 简单的线性变换（白化）操作，就可以达到BERT-flow的效果。
# 测试任务：GLUE的STS-B。
# 来源： https://github.com/bojone/BERT-whitening/blob/main/demo.py
# 数据集来源：
# https://github.com/pluto-junzeng/CNSD
# Chinese-STS-B：
# https://6a75-junzeng-uxxxm-1300734931.tcb.qcloud.la/STS-B.rar?sign=fa8d3ee7bc4e07d9ef64042f2d4f2465&t=1578114501
# https://spaces.ac.cn/archives/9079

import os
os.environ['TF_KERAS'] = '1'
USERNAME = os.getenv("USERNAME")
import numpy as np
import scipy.stats
from bert4keras.backend import keras, K
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.snippets import open, sequence_padding
from keras.models import Model


def load_train_data(filename):
    """加载训练数据（带标签）
    单条格式：(文本1, 文本2, 标签)
    """
    D = []
    with open(filename, encoding='utf-8') as f:
        for i, l in enumerate(f):
            if i > 0:
                l = l.strip().split('||')
                D.append((l[-3], l[-2], float(l[-1])))
    return D

# 加载数据集
datasets = {
    'sts-b-train': load_train_data(rf'D:\Users\{USERNAME}\data\STS-B\cnsd-sts-train.txt'),
    'sts-b-test': load_train_data(rf'D:\Users\{USERNAME}\data\STS-B\cnsd-sts-test.txt')
}

# bert配置
# config_path = '/root/kg/bert/uncased_L-12_H-768_A-12/bert_config.json'
# checkpoint_path = '/root/kg/bert/uncased_L-12_H-768_A-12/bert_model.ckpt'
# dict_path = '/root/kg/bert/uncased_L-12_H-768_A-12/vocab.txt'

# BERT base
config_path = rf'D:/Users/{USERNAME}/data/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = rf'D:/Users/{USERNAME}/data/chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = rf'D:/Users/{USERNAME}/data/chinese_L-12_H-768_A-12/vocab.txt'
# sts-b-train: 0.6761480432103012
# sts-b-test: 0.6862905169976401
# avg: 0.6812192801039707
# w-avg: 0.6782411789236216

# Robert配置
# https://github.com/ZhuiyiTechnology/pretrained-models
# https://open.zhuiyi.ai/releases/nlp/models/zhuiyi/chinese_roberta_L-4_H-312_A-12.zip
# config_path = rf'D:/Users/{USERNAME}/data/chinese_roberta_L-4_H-312_A-12/bert_config.json'
# checkpoint_path = rf'D:/Users/{USERNAME}/data/chinese_roberta_L-4_H-312_A-12/bert_model.ckpt'
# dict_path = rf'D:/Users/{USERNAME}/data/chinese_roberta_L-4_H-312_A-12/vocab.txt'
# sts-b-train: 0.6126753251814826
# sts-b-test: 0.6299551901382997
# avg: 0.6213152576598912
# w-avg: 0.616241427812935

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)  # 建立分词器


class GlobalAveragePooling1D(keras.layers.GlobalAveragePooling1D):
    """自定义全局池化
    """
    def call(self, inputs, mask=None):
        if mask is not None:
            mask = K.cast(mask, K.floatx())[:, :, None]
            return K.sum(inputs * mask, axis=1) / K.sum(mask, axis=1)
        else:
            return K.mean(inputs, axis=1)


# 建立模型
bert = build_transformer_model(config_path, checkpoint_path)

encoder_layers, count = [], 0
while True:
    try:
        output = bert.get_layer(
            'Transformer-%d-FeedForward-Norm' % count
        ).output
        encoder_layers.append(output)
        count += 1
    except:
        break

n_last, outputs = 2, []
for i in range(n_last):
    outputs.append(GlobalAveragePooling1D()(encoder_layers[-i]))

output = keras.layers.Average()(outputs)

# 最后的编码器
encoder = Model(bert.inputs, output)


def convert_to_vecs(data, maxlen=64):
    """转换文本数据为id形式
    """
    a_token_ids, b_token_ids, labels = [], [], []
    for d in data:
        token_ids = tokenizer.encode(d[0], maxlen=maxlen)[0]
        a_token_ids.append(token_ids)
        token_ids = tokenizer.encode(d[1], maxlen=maxlen)[0]
        b_token_ids.append(token_ids)
        labels.append(d[2])
    a_token_ids = sequence_padding(a_token_ids)
    b_token_ids = sequence_padding(b_token_ids)
    a_vecs = encoder.predict([a_token_ids,
                              np.zeros_like(a_token_ids)],
                             verbose=True)
    b_vecs = encoder.predict([b_token_ids,
                              np.zeros_like(b_token_ids)],
                             verbose=True)
    return a_vecs, b_vecs, np.array(labels)


def compute_kernel_bias(vecs, n_components=256):
    """计算kernel和bias
    vecs.shape = [num_samples, embedding_size]，
    最后的变换：y = (x + bias).dot(kernel)
    n_components: 最后向量输出的维度
    """
    vecs = np.concatenate(vecs, axis=0)
    mu = vecs.mean(axis=0, keepdims=True)
    cov = np.cov(vecs.T)
    u, s, vh = np.linalg.svd(cov)
    W = np.dot(u, np.diag(1 / np.sqrt(s)))
    # return None, None
    # return W, -mu
    return W[:, :n_components], -mu


def transform_and_normalize(vecs, kernel=None, bias=None):
    """应用变换，然后标准化
    输入 vecs.shape = (N, kernel.shape[0])
    return shape=(N, kernel.shape[1])
    """
    if not (kernel is None or bias is None):
        vecs = (vecs + bias).dot(kernel)
    return vecs / (vecs**2).sum(axis=1, keepdims=True)**0.5


def compute_corrcoef(x, y):
    """Spearman相关系数
    """
    return scipy.stats.spearmanr(x, y).correlation


# 语料向量化
all_names, all_weights, all_vecs, all_labels = [], [], [], []
for name, data in datasets.items():
    a_vecs, b_vecs, labels = convert_to_vecs(data)
    all_names.append(name)
    all_weights.append(len(data))
    all_vecs.append((a_vecs, b_vecs))
    all_labels.append(labels)

# 计算变换矩阵和偏置项
kernel, bias = compute_kernel_bias([v for vecs in all_vecs for v in vecs])
# kernel.shape, bias.shape
# Out[4]: ((312, 256), (1, 312))

# 变换，标准化，相似度，相关系数
all_corrcoefs = []
for (a_vecs, b_vecs), labels in zip(all_vecs, all_labels):
    a_vecs = transform_and_normalize(a_vecs, kernel, bias)
    b_vecs = transform_and_normalize(b_vecs, kernel, bias)
    sims = (a_vecs * b_vecs).sum(axis=1)
    corrcoef = compute_corrcoef(labels, sims)
    all_corrcoefs.append(corrcoef)

all_corrcoefs.extend([
    np.average(all_corrcoefs),
    np.average(all_corrcoefs, weights=all_weights)
])

for name, corrcoef in zip(all_names + ['avg', 'w-avg'], all_corrcoefs):
    print('%s: %s' % (name, corrcoef))

##########################################################################################################################
# 使用示例, 将一组32维的向量降低维度到8：
v_data = np.random.random(size=(10, 32))
kernel,bias=compute_kernel_bias(v_data, 8)
v_data2=transform_and_normalize(v_data, kernel=kernel, bias=bias)
# v_data2.shape
# Out[28]: (10, 8)
##########################################################################################################################

def main():
    pass


if __name__ == '__main__':
    main()
