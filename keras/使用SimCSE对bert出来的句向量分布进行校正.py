#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 资料来源：https://github.com/bojone/SimCSE

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

'''SimCSE是为了计算两个句子相似度，但构建句子相似度的数据很麻烦，SimCSE这里就采用了一个很简单但是非常有效的方式，就是同一个句子分别做dropout，把这当成一个正样本，其他句子当成负样本
第0位的样本1只和第1位的样本1是正样本，也就是说该位置的label为1，其他全为0
把上面的y_true转成数字0，1后，可以看到第0位的样本1的label就是
[0, 1, 0,..., 0,0,0]
以此类推，这就是我们自己构造出来的y_true
'''
# keras.__version__
# Out[29]: '2.6.0'
# tf.__version__
# Out[30]: '2.6.0'
# keras_bert.__version__
# Out[31]: '0.89.0'

config_path = r'D:\Users\{}\data\RoBERTa-tiny-clue\bert_config.json'.format(os.getenv("USERNAME"))
checkpoint_path = r'D:\Users\{}\data\RoBERTa-tiny-clue\bert_model.ckpt'.format(os.getenv("USERNAME"))
dict_path = r'D:\Users\{}\data\RoBERTa-tiny-clue\vocab.txt'.format(os.getenv("USERNAME"))


token_dict = {}
with open(dict_path, 'r', encoding='utf-8') as f:
    for line in f.readlines():
        token = line.strip()
        token_dict[token] = len(token_dict)

tokenizer = Tokenizer(token_dict)
cut_words = tokenizer.tokenize(u'今天天气不错')
print(cut_words)

bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=None)

def get_features(text_list):
    '''获取文本特征'''
    vec_list = []
    for text in text_list:
        indices, segments = tokenizer.encode(text, second=None, max_len=8)
        predicts = bert_model.predict([np.array([indices]), np.array([segments])])[0]
        vec = predicts[0].tolist()
        vec_list.append(vec)
    return vec_list


model_type = 'RoBERTa'
pooling = 'mean'
task_name = 'LCQMC'
dropout_rate = 0.5
maxlen = 64
# - model_type: 模型，必须是['BERT', 'RoBERTa', 'WoBERT', 'RoFormer', 'BERT-large', 'RoBERTa-large', 'SimBERT', 'SimBERT-tiny', 'SimBERT-small']之一；
# - pooling: 池化方式，必须是['first-last-avg', 'last-avg', 'cls', 'pooler']之一；
# - task_name: 评测数据集，必须是['ATEC', 'BQ', 'LCQMC', 'PAWSX', 'STS-B']之一；
# - dropout_rate: 浮点数，dropout的比例，如果为0则不dropout；


def get_encoder(
    config_path,
    checkpoint_path,
    pooling='first-last-avg',
):
    """建立编码器
    """
    bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=None)
    for l in bert_model.layers[-4:]:
        l.trainable = True
    outputs, count = [], 1
    while True:
        try:
            output = bert_model.get_layer(
                'Encoder-%d-FeedForward-Norm' % count
            ).output
            outputs.append(output)
            count += 1
        except:
            break

    if pooling == 'first-last-avg':
        outputs = [
            keras.layers.GlobalAveragePooling1D()(outputs[0]),
            keras.layers.GlobalAveragePooling1D()(outputs[-1])
        ]
        output = keras.layers.Average()(outputs)
    elif pooling == 'last-avg':
        output = keras.layers.GlobalAveragePooling1D()(outputs[-1])
    elif pooling == 'cls':
        output = keras.layers.Lambda(lambda x: x[:, 0])(outputs[-1])
    elif pooling == 'mean':
        output = keras.layers.Lambda(lambda x: K.mean(x, axis=1) )(outputs[-1])
    elif pooling == 'pooler':
        output = bert_model.outputs
    else:
        output = bert_model.outputs

    # 最后的编码器
    encoder = Model(bert_model.inputs, output)
    return encoder


encoder = get_encoder(
        config_path,
        checkpoint_path,
        pooling=pooling,
    )

def load_data(filename):
    """加载数据（带标签）
    单条格式：(文本1, 文本2, 标签)
    """
    D = []
    df = pd.read_csv(filename, sep='\t', encoding='utf-8')
    for text_a, text_b, label in df[ ['text_a', 'text_b', 'label']].values:
        D.append((text_a, text_b, float(label)))
    return D

USERNAME = os.getenv("USERNAME")
data_path = rf'D:\Users\{USERNAME}\github_project\lcqmc_data'
datasets = {
    '%s-%s' % (task_name, f):
    load_data('%s/%s.txt' % (data_path, f))
    for f in ['train', 'dev', 'test']
}

def convert_to_ids(data, tokenizer, maxlen=64):
    """转换文本数据为id形式
    """
    a_token_ids, b_token_ids, labels = [], [], []
    for d in tqdm(data):
        token_ids = tokenizer.encode(d[0], max_len=maxlen)[0]
        a_token_ids.append(token_ids)
        token_ids = tokenizer.encode(d[1], max_len=maxlen)[0]
        b_token_ids.append(token_ids)
        labels.append(d[2])
    a_token_ids = np.array(a_token_ids)
    b_token_ids = np.array(b_token_ids)
    return a_token_ids, b_token_ids, labels

# 语料id化
all_names, all_weights, all_token_ids, all_labels = [], [], [], []
train_token_ids = []
for name, data in datasets.items():
    a_token_ids, b_token_ids, labels = convert_to_ids(data, tokenizer, maxlen)
    all_names.append(name)
    all_weights.append(len(data))
    all_token_ids.append((a_token_ids, b_token_ids))
    all_labels.append(labels)
    train_token_ids.extend(a_token_ids)
    train_token_ids.extend(b_token_ids)

class DataGenerator(object):
    """数据生成器模版
    """
    def __init__(self, data, batch_size=32, buffer_size=None):
        self.data = data
        self.batch_size = batch_size
        if hasattr(self.data, '__len__'):
            self.steps = len(self.data) // self.batch_size
            if len(self.data) % self.batch_size != 0:
                self.steps += 1
        else:
            self.steps = None
        self.buffer_size = buffer_size or batch_size * 1000

    def __len__(self):
        return self.steps

    def sample(self, random=False):
        """采样函数，每个样本同时返回一个is_end标记
        """
        if random:
            if self.steps is None:

                def generator():
                    caches, isfull = [], False
                    for d in self.data:
                        caches.append(d)
                        if isfull:
                            i = np.random.randint(len(caches))
                            yield caches.pop(i)
                        elif len(caches) == self.buffer_size:
                            isfull = True
                    while caches:
                        i = np.random.randint(len(caches))
                        yield caches.pop(i)

            else:

                def generator():
                    for i in np.random.permutation(len(self.data)):
                        yield self.data[i]

            data = generator()
        else:
            data = iter(self.data)

        d_current = next(data)
        for d_next in data:
            yield False, d_current
            d_current = d_next

        yield True, d_current

    def __iter__(self, random=False):
        raise NotImplementedError

    def forfit(self, random=True):
        while True:
            for d in self.__iter__(random):
                yield d

    def fortest(self, random=False):
        while True:
            for d in self.__iter__(random):
                yield d[0]

    def to_dataset(self, types, shapes, names=None, padded_batch=False):
        """转为tf.data.Dataset格式
        如果传入names的话，自动把数据包装成dict形式。
        """
        if names is None:

            generator = self.forfit

        else:

            if is_string(names):
                warps = lambda k, v: {k: v}
            elif is_string(names[0]):
                warps = lambda k, v: dict(zip(k, v))
            else:
                warps = lambda k, v: tuple(
                    dict(zip(i, j)) for i, j in zip(k, v)
                )

            def generator():
                for d in self.forfit():
                    yield warps(names, d)

            types = warps(names, types)
            shapes = warps(names, shapes)

        if padded_batch:
            dataset = tf.data.Dataset.from_generator(
                generator, output_types=types
            )
            dataset = dataset.padded_batch(self.batch_size, shapes)
        else:
            dataset = tf.data.Dataset.from_generator(
                generator, output_types=types, output_shapes=shapes
            )
            dataset = dataset.batch(self.batch_size)

        return dataset


class data_generator(DataGenerator):
    """训练语料生成器
    """
    def __iter__(self, random=False):
        batch_token_ids = []
        for is_end, token_ids in self.sample(random):
            batch_token_ids.append(token_ids)
            batch_token_ids.append(token_ids)  #  每个batch内，每一句话都重复了一次。
            if len(batch_token_ids) == self.batch_size * 2 or is_end:
                batch_token_ids = np.array(batch_token_ids)
                batch_segment_ids = np.zeros_like(batch_token_ids)
                batch_labels = np.zeros_like(batch_token_ids[:, :1])
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids = []

def simcse_loss(y_true, y_pred):
    """用于SimCSE训练的loss
    首先，输入的y_true，y_pred是句向量（这里的y_true是用不到的，只用到y_pred）
    SimCSE是为了计算两个句子相似度，但构建句子相似度的数据很麻烦，SimCSE这里就采用了一个很简单但是非常有效的方式，就是同一个句子分别做dropout，把这当成一个正样本，其他句子当成负样本
    第0位的样本1只和第1位的样本1是正样本，也就是说该位置的label为1，其他全为0
    把上面的y_true转成数字0，1后，可以看到第0位的样本1的label就是
    [0, 1, 0,..., 0,0,0]
    以此类推，这就是我们自己构造出来的y_true

    每个batch内，每一句话都重复了一次。举例来说，句子a，b，c。编成一个batch就是：[a，a，b，b，c，c]。
    这个loss的输入中y_true只是凑数的，并不起作用。因为真正的y_true是通过batch内数据计算得出的。y_pred就是batch内的每句话的embedding，通过bert编码得来。

    """
    # 构造标签
    idxs = K.arange(0, K.shape(y_pred)[0])
    idxs_1 = idxs[None, :]
    idxs_2 = (idxs + 1 - idxs % 2 * 2)[:, None]  # 如果一个句子id为奇数，那么和它同义的句子的id就是它的上一句，如果一个句子id为偶数，那么和它同义的句子的id就是它的下一句。 [:, None] 是在列上添加一个维度。
    y_true = K.equal(idxs_1, idxs_2)
    y_true = K.cast(y_true, K.floatx())
    # 计算相似度
    y_pred = K.l2_normalize(y_pred, axis=1)
    similarities = K.dot(y_pred, K.transpose(y_pred))
    similarities = similarities - tf.eye(K.shape(y_pred)[0]) * 1e12
    similarities = similarities * 20  # 将所有相似度乘以20，这个目的是想计算softmax概率时，更加有区分度。
    loss = K.categorical_crossentropy(y_true, similarities, from_logits=True)
    return K.mean(loss)

# SimCSE训练
encoder.summary()
encoder.compile(loss=simcse_loss, optimizer=Adam(1e-5))
train_generator = data_generator(train_token_ids, 64)
encoder.fit(
    train_generator.forfit(), steps_per_epoch=len(train_generator), epochs=1
)

encoder.save(r'D:\Users\{}\data\SimCSE-bert-base\My_SimCSE.h5'.format(os.getenv("USERNAME")))

custom_objects = {layer.__class__.__name__:layer for layer in bert_model.layers}
custom_objects['simcse_loss'] = simcse_loss
encoder = load_model(r'D:\Users\{}\data\SimCSE-bert-base\My_SimCSE.h5'.format(os.getenv("USERNAME")), custom_objects=custom_objects)

def l2_normalize(vecs):
    """标准化
    """
    norms = (vecs**2).sum(axis=1, keepdims=True)**0.5
    return vecs / np.clip(norms, 1e-8, np.inf)

# 将校正后的向量可视化
vis_word = ["中国", "市场", "公司", "美国", "记者", "学生", "游戏", "北京",
           "投资", "电影", "银行", "工作", "留学", "大学", "经济", "产品",
           "设计", "方面", "玩家", "学校", "学习", "房价", "专家", "楼市"]

a_token_ids = []
for word in tqdm(vis_word):
    token_ids = tokenizer.encode(word, max_len=maxlen)[0]
    a_token_ids.append(token_ids)
a_token_ids = np.array(a_token_ids)
a_vecs = encoder.predict([a_token_ids,
                          np.zeros_like(a_token_ids)],
                         verbose=True)
word_vec = l2_normalize(a_vecs)

# 降维+可视化
# tsne = TSNE(n_components=2, random_state=123)
# word2vec_tsne = tsne.fit_transform(np.array(word_vec))
pca=PCA(n_components=2)
word2vec_tsne=pca.fit_transform(np.array(word_vec))

# 查询这些词在词表中的索引（序号）
# vis_word_idx = [words.index(word) for words in vis_word]

plt.figure(figsize=[10, 8])
for index, word in enumerate(vis_word):
    plt.scatter(word2vec_tsne[index, 0], word2vec_tsne[index, 1])
    plt.text(word2vec_tsne[index, 0], word2vec_tsne[index, 1], word)

plt.grid(True)
plt.title("SimCSE-BERT校正部分词向量的分布情况", fontsize=15)
save_image_name = "SimCSE-BERT校正部分词向量的分布情况.jpg"
plt.savefig(save_image_name, dpi=500,bbox_inches = 'tight')
plt.show()

# 466 / 8128[ > .............................] - ETA: 4:30: 44 - loss: 2.0683
# 2807/8128 [=========>....................] - ETA: 3:10:05 - loss: 2.0654

# 公开的 SimCSE-bert模型： https://huggingface.co/tuhailong/SimCSE-bert-base/tree/main
# 经验证，其表示句向量效果的确好于原始的bert；

def main():
    pass


if __name__ == '__main__':
    main()

