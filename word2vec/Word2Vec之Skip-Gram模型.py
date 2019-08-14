#!/usr/bin/python3
# coding: utf-8

# Word2Vec之Skip-Gram模型

import time
import numpy as np
import tensorflow as tf
import random
from collections import Counter

# https://github.com/NELSONZHAO/zhihu/blob/master/skip_gram/data/Javasplittedwords
# https://github.com/NELSONZHAO/zhihu/blob/master/skip_gram/Skip-Gram-Chinese-Corpus.ipynb

with open('/home/gswyhq/github_projects/zhihu/skip_gram/data/Javasplittedwords') as f:
    text = f.read()

words = text.split()

# 数据预处理
# 数据预处理过程主要包括：
#
# 替换文本中特殊符号并去除低频词
# 对文本分词
# 构建语料
# 单词映射表
# 筛选低频词
words_count = Counter(words)
words = [w for w in words if words_count[w] > 50]

# 构建映射表
vocab = set(words)
vocab_to_int = {w: c for c, w in enumerate(vocab)}
int_to_vocab = {c: w for c, w in enumerate(vocab)}

print("total words: {}".format(len(words)))
print("unique words: {}".format(len(set(words))))

# 对原文本进行vocab到int的转换
int_words = [vocab_to_int[w] for w in words]

# 采样
# 对停用词进行采样，例如“the”， “of”以及“for”这类单词进行剔除。剔除这些单词以后能够加快我们的训练过程，同时减少训练过程中的噪音。


t = 1e-5 # t值
threshold = 0.9 # 剔除概率阈值

# 统计单词出现频次
int_word_counts = Counter(int_words)
total_count = len(int_words)
# 计算单词频率
word_freqs = {w: c/total_count for w, c in int_word_counts.items()}
# 计算被删除的概率
prob_drop = {w: 1 - np.sqrt(t / word_freqs[w]) for w in int_word_counts}
# 对单词进行采样
train_words = [w for w in int_words if prob_drop[w] < threshold]

# 构造batch
# Skip - Gram模型是通过输入词来预测上下文。因此我们要构造我们的训练样本，具体思想请参考知乎专栏，这里不再重复。
#
# 对于一个给定词，离它越近的词可能与它越相关，离它越远的词越不相关，这里我们设置窗口大小为5，对于每个训练单词，我们还会在[1:5]
# 之间随机生成一个整数R，用R作为我们最终选择output
# word的窗口大小。这里之所以多加了一步随机数的窗口重新选择步骤，是为了能够让模型更聚焦于当前input
# word的邻近词。

def get_targets(words, idx, window_size=5):
    '''
    获得input word的上下文单词列表

    参数
    ---
    words: 单词列表
    idx: input word的索引号
    window_size: 窗口大小
    '''
    target_window = np.random.randint(1, window_size + 1)
    # 这里要考虑input word前面单词不够的情况
    start_point = idx - target_window if (idx - target_window) > 0 else 0
    end_point = idx + target_window
    # output words(即窗口中的上下文单词)
    targets = set(words[start_point: idx] + words[idx + 1: end_point + 1])
    return list(targets)


def get_batches(words, batch_size, window_size=5):
    '''
    构造一个获取batch的生成器
    '''
    n_batches = len(words) // batch_size

    # 仅取full batches
    words = words[:n_batches * batch_size]

    for idx in range(0, len(words), batch_size):
        x, y = [], []
        batch = words[idx: idx + batch_size]
        for i in range(len(batch)):
            batch_x = batch[i]
            batch_y = get_targets(batch, i, window_size)
            # 由于一个input word会对应多个output word，因此需要长度统一
            x.extend([batch_x] * len(batch_y))
            y.extend(batch_y)
        yield x, y



# 构建网络
# 该部分主要包括：
#
# 输入层
# Embedding
# Negative
# Sampling

# 输入
train_graph = tf.Graph()
with train_graph.as_default():
    inputs = tf.placeholder(tf.int32, shape=[None], name='inputs')
    labels = tf.placeholder(tf.int32, shape=[None, None], name='labels')

# Embedding
# 嵌入矩阵的矩阵形状为 vocab_size×hidden_units_size
#
# TensorFlow中的tf.nn.embedding_lookup函数可以实现lookup的计算方式


vocab_size = len(int_to_vocab)
embedding_size = 200  # 嵌入维度

with train_graph.as_default():
    # 嵌入层权重矩阵
    embedding = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1, 1))
    # 实现lookup
    embed = tf.nn.embedding_lookup(embedding, inputs)

# Negative Sampling
# 负采样主要是为了解决梯度下降计算速度慢的问题，详情同样参考我的上一篇知乎专栏文章。
#
# TensorFlow中的tf.nn.sampled_softmax_loss会在softmax层上进行采样计算损失，计算出的loss要比full softmax loss低。

n_sampled = 100

with train_graph.as_default():
    softmax_w = tf.Variable(tf.truncated_normal([vocab_size, embedding_size], stddev=0.1))
    softmax_b = tf.Variable(tf.zeros(vocab_size))

    # 计算negative sampling下的损失
    loss = tf.nn.sampled_softmax_loss(softmax_w, softmax_b, labels, embed, n_sampled, vocab_size)

    cost = tf.reduce_mean(loss)
    optimizer = tf.train.AdamOptimizer().minimize(cost)

# 验证
# 为了更加直观的看到我们训练的结果，我们将查看训练出的相近语义的词。


with train_graph.as_default():
    # 随机挑选一些单词
    ## From Thushan Ganegedara's implementation
    valid_size = 7  # Random set of words to evaluate similarity on.
    valid_window = 100
    # pick 8 samples from (0,100) and (1000,1100) each ranges. lower id implies more frequent
    valid_examples = np.array(random.sample(range(valid_window), valid_size // 2))
    valid_examples = np.append(valid_examples,
                               random.sample(range(1000, 1000 + valid_window), valid_size // 2))
    valid_examples = [vocab_to_int['word'],
                      vocab_to_int['ppt'],
                      vocab_to_int['熟悉'],
                      vocab_to_int['java'],
                      vocab_to_int['能力'],
                      vocab_to_int['逻辑思维'],
                      vocab_to_int['了解']]

    valid_size = len(valid_examples)
    # 验证单词集
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    # 计算每个词向量的模并进行单位化
    norm = tf.sqrt(tf.reduce_sum(tf.square(embedding), 1, keep_dims=True))
    normalized_embedding = embedding / norm
    # 查找验证单词的词向量
    valid_embedding = tf.nn.embedding_lookup(normalized_embedding, valid_dataset)
    # 计算余弦相似度
    similarity = tf.matmul(valid_embedding, tf.transpose(normalized_embedding))

epochs = 10  # 迭代轮数
batch_size = 1000  # batch大小
window_size = 10  # 窗口大小

with train_graph.as_default():
    saver = tf.train.Saver()  # 文件存储

with tf.Session(graph=train_graph) as sess:
    iteration = 1
    loss = 0
    sess.run(tf.global_variables_initializer())

    for e in range(1, epochs + 1):
        batches = get_batches(train_words, batch_size, window_size)
        start = time.time()
        #
        for x, y in batches:

            feed = {inputs: x,
                    labels: np.array(y)[:, None]}
            train_loss, _ = sess.run([cost, optimizer], feed_dict=feed)

            loss += train_loss

            if iteration % 100 == 0:
                end = time.time()
                print("Epoch {}/{}".format(e, epochs),
                      "Iteration: {}".format(iteration),
                      "Avg. Training loss: {:.4f}".format(loss / 100),
                      "{:.4f} sec/batch".format((end - start) / 100))
                loss = 0
                start = time.time()

            # 计算相似的词
            if iteration % 1000 == 0:
                # 计算similarity
                sim = similarity.eval()
                for i in range(valid_size):
                    valid_word = int_to_vocab[valid_examples[i]]
                    top_k = 8  # 取最相似单词的前8个
                    nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                    log = 'Nearest to [%s]:' % valid_word
                    for k in range(top_k):
                        close_word = int_to_vocab[nearest[k]]
                        log = '%s %s,' % (log, close_word)
                    print(log)

            iteration += 1

    save_path = saver.save(sess, "checkpoints/text8.ckpt")
    embed_mat = sess.run(normalized_embedding)

