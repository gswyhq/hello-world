#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 资料来源：https://github.com/bojone/CoSENT

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
import scipy
mpl.rcParams['font.sans-serif'] = ['SimHei'] #指定默认字体   
mpl.rcParams['axes.unicode_minus'] = False #解决保存图像是负号'-'显示为方块的问题

'''标准的SimCSE是只需要正样本对的（通过Dropout或者人工标注构建），然后它将batch内的所有其他样本都视为负样本；
而有监督版的SimCSE则是需要三元组的数据，它实际上就是把困难样本补充到标准的SimCSE上，
即负样本不只有batch内的所有其他样本，还有标注的困难样本，但同时正样本依然不能缺，所以需要“(原始句子, 相似句子, 不相似句子)”的三元组数据。

至于CoSENT，它只用到了标注好的正负样本对，也不包含随机采样batch内的其他样本来构建负样本的过程，
我们也可以将它理解为对比学习，但它是“样本对”的对比学习，而不是像SimCSE的“样本”对比学习，也就是说，它的“单位”是一对句子而不是一个句子。
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
batch_size=32
epochs=1
# - model_type: 模型，必须是['BERT', 'RoBERTa', 'WoBERT', 'RoFormer', 'BERT-large', 'RoBERTa-large', 'SimBERT', 'SimBERT-tiny', 'SimBERT-small']之一；
# - pooling: 池化方式，必须是['first-last-avg', 'last-avg', 'cls', 'pooler']之一；
# - task_name: 评测数据集，必须是['ATEC', 'BQ', 'LCQMC', 'PAWSX', 'STS-B']之一；
# - dropout_rate: 浮点数，dropout的比例，如果为0则不dropout；


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
datasets = [
    load_data('%s/%s.txt' % (data_path, f))
    for f in ['train', 'dev', 'test']
]
train_data, valid_data, test_data = datasets

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

class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text1, text2, label) in self.sample(random):
            for text in [text1, text2]:
                token_ids, segment_ids = tokenizer.encode(text, max_len=maxlen)
                batch_token_ids.append(token_ids)
                batch_segment_ids.append(segment_ids)
                batch_labels.append([label])
            if len(batch_token_ids) == self.batch_size * 2 or is_end:
                batch_token_ids = np.array(batch_token_ids)
                batch_segment_ids = np.array(batch_segment_ids)
                batch_labels = np.array(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


# 转换数据集
train_generator = data_generator(train_data, batch_size)
valid_generator = data_generator(valid_data, batch_size)
test_generator = data_generator(test_data, batch_size)


def cosent_loss(y_true, y_pred):
    """排序交叉熵
    y_true：标签/打分，y_pred：句向量
    """
    y_true = y_true[::2, 0]  # 获取偶数位标签，即取出真实的标签；
    y_true = K.cast(y_true[:, None] < y_true[None, :], K.floatx())  # 取出负例-正例的差值
    y_pred = K.l2_normalize(y_pred, axis=1)  # 对输出的句子向量进行l2归一化   后面只需要对应位相乘  就可以得到cos值了
    y_pred = K.sum(y_pred[::2] * y_pred[1::2], axis=1) * 20  # 奇偶位向量相乘，得到对应cos
    y_pred = y_pred[:, None] - y_pred[None, :]  # 取出负例-正例的差值, # 这里是算出所有位置 两两之间余弦的差值
    # 矩阵中的第i行j列  表示的是第i个余弦值-第j个余弦值
    y_pred = K.reshape(y_pred - (1 - y_true) * 1e12, [-1])  # 乘以e的12次方,要排除掉不需要计算(mask)的部分
    y_pred = K.concatenate([[0], y_pred], axis=0)  # 这里加0是因为e^0 = 1相当于在log中加了1
    return K.logsumexp(y_pred)


# 构建模型
# base = build_transformer_model(config_path, checkpoint_path)
# bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=None)
for l in bert_model.layers[-4:]:
    l.trainable = True
# output = keras.layers.Lambda(lambda x: x[:, 0])(base.output)
# output = keras.layers.GlobalAveragePooling1D()(base.output)
output = keras.layers.Lambda(lambda x: K.mean(x, axis=1))(bert_model.get_layer(
            'Encoder-4-FeedForward-Norm').output)
encoder = keras.models.Model(bert_model.inputs, output)
model = encoder
model.compile(loss=cosent_loss, optimizer=Adam(2e-5))


def compute_corrcoef(x, y):
    """Spearman相关系数
    """
    return scipy.stats.spearmanr(x, y).correlation


def l2_normalize(vecs):
    """l2标准化
    """
    norms = (vecs**2).sum(axis=1, keepdims=True)**0.5
    return vecs / np.clip(norms, 1e-8, np.inf)

save_h5_file = r'D:\Users\{}\data\SimCSE-bert-base\My_CoSENT.h5'.format(os.getenv("USERNAME"))
class Evaluator(keras.callbacks.Callback):
    """保存验证集分数最好的模型
    """
    def __init__(self):
        self.best_val_score = 0.

    def on_epoch_end(self, epoch, logs=None):
        val_score = self.evaluate(valid_generator)
        if val_score > self.best_val_score:
            self.best_val_score = val_score
            model.save(save_h5_file)
        print(
            u'val_score: %.5f, best_val_score: %.5f\n' %
            (val_score, self.best_val_score)
        )

    def evaluate(self, data):
        Y_true, Y_pred = [], []
        for x_true, y_true in data:
            Y_true.extend(y_true[::2, 0])
            x_vecs = encoder.predict(x_true)
            x_vecs = l2_normalize(x_vecs)
            y_pred = (x_vecs[::2] * x_vecs[1::2]).sum(1)
            Y_pred.extend(y_pred)
        return compute_corrcoef(Y_true, Y_pred)

evaluator = Evaluator()

model.fit_generator(
    train_generator.forfit(),
    steps_per_epoch=len(train_generator),
    epochs=epochs,
    callbacks=[evaluator]
)
# model.load_weights('%s.cosent.weights' % task_name)
test_score = evaluator.evaluate(test_generator)
print(u'test_score: %.5f' % test_score)


###########################################################################################################################
custom_objects = {layer.__class__.__name__:layer for layer in bert_model.layers}
custom_objects['cosent_loss'] =cosent_loss
encoder2 = load_model(save_h5_file, custom_objects=custom_objects)

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
a_vecs = encoder2.predict([a_token_ids,
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
plt.title("CoSENT-BERT校正部分词向量的分布情况", fontsize=15)
save_image_name = "CoSENT-BERT校正部分词向量的分布情况.jpg"
plt.savefig(save_image_name, dpi=500,bbox_inches = 'tight')
plt.show()


def main():
    pass


if __name__ == '__main__':
    main()

