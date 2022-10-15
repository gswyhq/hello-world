#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import numpy as np
import math
from keras.layers import Input, Lambda, Dense
from keras.models import Model, load_model
import keras.backend as K
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
from tensorflow.keras.utils import Sequence
from keras.callbacks import History
from sklearn.manifold import TSNE

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei'] #指定默认字体   
mpl.rcParams['axes.unicode_minus'] = False #解决保存图像是负号'-'显示为方块的问题

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
        # 使用预训练好的 Bert 直接获得句子向量，可以是 CLS 位的向量，也可以是不同 token 向量的平均值。
        # vec = predicts[0].tolist()  # 取一号位(CLS位)作为整个句子向量；
        vec = predicts.mean(axis=0).tolist()  # 取均值作为整个句子向量；通过小量数据测试,均值向量效果好于CLS位向量；
        vec_list.append(vec)
    return vec_list

vis_word = ["中国", "市场", "公司", "美国", "记者", "学生", "游戏", "北京",
           "投资", "电影", "银行", "工作", "留学", "大学", "经济", "产品",
           "设计", "方面", "玩家", "学校", "学习", "房价", "专家", "楼市"]

word_vec = get_features(vis_word)


# TSNE降维+可视化
# tsne = TSNE(n_components=2, random_state=123)
# word2vec_tsne = tsne.fit_transform(np.array(word_vec))

# PCA降维
pca=PCA(n_components=2)
word2vec_tsne=pca.fit_transform(np.array(word_vec))

# 查询这些词在词表中的索引（序号）
# vis_word_idx = [words.index(word) for words in vis_word]

plt.figure(figsize=[10, 8])
for index, word in enumerate(vis_word):
    plt.scatter(word2vec_tsne[index, 0], word2vec_tsne[index, 1])
    plt.text(word2vec_tsne[index, 0], word2vec_tsne[index, 1], word)

plt.grid(True)
plt.title("部分词向量的分布情况", fontsize=15)
plt.savefig("部分词向量的分布情况", dpi=500,bbox_inches = 'tight')
plt.show()

# 通过上面结果，可以看出：对于Bert输出的向量直接用于相似度计算的问题上，往往表现是较差的。
# 语言模型的目标就是最大化token与上下文的共现概率，在该目标下， 上下文c 与 建模单词x 的表示会不断的拉近，如果同一个 建模单词x 存在于另一个上下文 c' 中，那么在训练中 上下文c 与 上下文c' 的表示也会不断拉近。
# 通过上述分析我们可以得出Bert的预训练过程和语义相似度的计算目标是十分接近的，训练得到的句向量包含了文本间的语义相似度的信息，原则上是可以通过点积(cosine)来进行相似度计算的。

# 实际操作时候效果往往不理想的原因呢？
# Bert模型的向量表示存在的问题
# 1.Bert的词向量在空间中的分布并不均匀
# Bert的词向量在空间中的分布呈现锥形，高频的词都靠近原点。 (所有词的均值)，而低频词远离原点，这会导致即使一个高频词与一个低频词的语义等价，
# 但是词频的差异也会带来巨大的差异，从而词向量的距离不能很好的表达词间的语义相关性。

# 2.高频词分布紧凑，低频词的分布稀疏
# 分布稀疏会导致区间的语义不完整(poorly defined)，低频词表示训练的不充分，而句向量仅仅是词向量的平均池化，所以计算出来的相似度存在问题。

# 故而需要校正BERT出来的句向量的分布，从而使得计算出来的cos相似度更为合理一些。
# 还有个原因：单词级相似度比较不适用于BERT embeddings，因为这些嵌入是上下文相关的，这意味着单词vector会根据它出现在的句子而变化。

########################################################################################################################
# 使用 SimCSE-bert-base 可视化
# SimCSE 通过对比学习的方法进行训练，并且可以用于无监督训练。在无监督的时候，SimCSE 会把通过不同的 Dropout 得到一个句子的两个向量，作为正样本对，而语料库里的其他句子向量可作为负样本。
# 模型来源：
# https://huggingface.co/tuhailong/SimCSE-bert-base/tree/main
from transformers import AutoTokenizer, AutoModel
from transformers import BertModel, BertTokenizer
model = AutoModel.from_pretrained(r'D:\Users\{}\data\SimCSE-bert-base'.format(os.getenv("USERNAME")))
# bert = BertModel.from_pretrained(r'D:\Users\{}\data\SimCSE-bert-base'.format(os.getenv("USERNAME")), from_tf=False) # mv optimizer.pt pytorch_model.bin
tokenizer = AutoTokenizer.from_pretrained(r'D:\Users\{}\data\SimCSE-bert-base'.format(os.getenv("USERNAME")))
sentences_str_list = ["今天天气不错的","天气不错的"]
inputs = tokenizer(sentences_str_list,return_tensors="pt", padding='max_length', truncation=True, max_length=32)
outputs = model(**inputs)

inputs = tokenizer(vis_word,return_tensors="pt", padding='max_length', truncation=True, max_length=32)
outputs = model(**inputs)
word_vec = np.array([t.data.numpy().mean(axis=0) for t in outputs[0]])

# 降维+可视化
# tsne = TSNE(n_components=2, random_state=123)
# word2vec_tsne = tsne.fit_transform(np.array(word_vec))

# PCA降维
pca=PCA(n_components=2)
word2vec_tsne=pca.fit_transform(np.array(word_vec))

plt.figure(figsize=[10, 8])
for index, word in enumerate(vis_word):
    plt.scatter(word2vec_tsne[index, 0], word2vec_tsne[index, 1])
    plt.text(word2vec_tsne[index, 0], word2vec_tsne[index, 1], word)

plt.grid(True)
plt.title("SimCSE-BERT部分词向量的分布情况", fontsize=15)
save_image_name = "SimCSE-BERT部分词向量的分布情况.jpg"
plt.savefig(save_image_name, dpi=500,bbox_inches = 'tight')
plt.show()

# 通过结果可以看出，使用SimCSE-BERT的结果明显好于使用原始BERT的结果；需要注意的是，降维方法用PCA，不能使用TSNE；句向量根据均值取，而不能取一号位；

########################################################################################################################
# BERT-whitening
# Bert-whitening，用预训练 Bert 获得所有句子的向量，得到句子向量矩阵，然后通过一个线性变换把句子向量矩阵变为一个均值 0，协方差矩阵为单位阵的矩阵。
def compute_kernel_bias(vecs, n_components=256):
    """计算kernel和bias
    vecs.shape = [num_samples, embedding_size]，
    最后的变换：y = (x + bias).dot(kernel)
    n_components: 转换后的维度，SVD出来的对角矩阵Λ已经从大到小排好序了，所以我们只需要保留前面若干维，就可以到达这个降维效果；这个操作其实就是PCA
    """
    mu = vecs.mean(axis=0, keepdims=True)
    cov = np.cov(vecs.T)
    u, s, vh = np.linalg.svd(cov)
    W = np.dot(u, np.diag(1 / np.sqrt(s)))
    return W[:, :n_components], -mu

kernel, bias = compute_kernel_bias(np.array(word_vec), n_components=2)
x = np.array(word_vec)
y = (x + bias).dot(kernel)

print("输出均值: {}\n协方差:{}".format(y.mean(), np.cov(y)))

# 降维+可视化
word2vec_tsne = y

# 查询这些词在词表中的索引（序号）
# vis_word_idx = [words.index(word) for words in vis_word]

plt.figure(figsize=[10, 8])
for index, word in enumerate(vis_word):
    plt.scatter(word2vec_tsne[index, 0], word2vec_tsne[index, 1])
    plt.text(word2vec_tsne[index, 0], word2vec_tsne[index, 1], word)

plt.grid(True)
plt.title("BERT-whitening部分词向量的分布情况", fontsize=15)
plt.savefig("BERT-whitening部分词向量的分布情况", dpi=500,bbox_inches = 'tight')
plt.show()

# BERT-whitening 的结果，跟原始结果无差异，可能是使用的方法不对，未用全量数据；
# 且效果跟降维方法有关；即在这里PCA降维效果看起来好于TSNE降维效果；

########################################################################################################################
# 使用 Sentence-BERT 可视化
# 有监督的方式主要是 Sentence-Bert (SBERT)，SBERT 通过 Bert 的孪生网络获得两个句子的向量，进行有监督学习
# 模型来源：
# https://huggingface.co/uer/sbert-base-chinese-nli/tree/main
from transformers import AutoTokenizer, AutoModel
from transformers import BertModel, BertTokenizer
model = AutoModel.from_pretrained(r'D:\Users\{}\data\sbert-base-chinese-nli'.format(os.getenv("USERNAME")))
tokenizer = AutoTokenizer.from_pretrained(r'D:\Users\{}\data\sbert-base-chinese-nli'.format(os.getenv("USERNAME")))
inputs = tokenizer(vis_word,return_tensors="pt", padding='max_length', truncation=True, max_length=32)
outputs = model(**inputs)
word_vec = np.array([t.data.numpy().mean(axis=0) for t in outputs[0]])

# PCA降维
pca=PCA(n_components=2)
word2vec_tsne=pca.fit_transform(np.array(word_vec))

plt.figure(figsize=[10, 8])
for index, word in enumerate(vis_word):
    plt.scatter(word2vec_tsne[index, 0], word2vec_tsne[index, 1])
    plt.text(word2vec_tsne[index, 0], word2vec_tsne[index, 1], word)

plt.grid(True)
plt.title("Sentence-BERT部分词向量的分布情况", fontsize=15)
save_image_name = "Sentence-BERT部分词向量的分布情况.jpg"
plt.savefig(save_image_name, dpi=500,bbox_inches = 'tight')
plt.show()


# 通过结果也可以看到 Sentence-BERT 的结果，要好于原始的Robert的结果；

def main():
    pass

if __name__ == '__main__':
    main()