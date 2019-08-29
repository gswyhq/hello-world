#!/usr/bin/env python
# coding: utf-8

# 来源： https://github.com/fchollet/deep-learning-with-python-notebooks.gi
# In[1]:


import keras
keras.__version__


# 单词和字符的 one-hot 编码
# 
# one-hot 编码是将标记转换为向量的最常用、最基本的方法。在第 3 章的 IMDB 和路透社两
# 个例子中,你已经用过这种方法(都是处理单词)
# 。它将每个单词与一个唯一的整数索引相关联,
# 然后将这个整数索引 i 转换为长度为 N 的二进制向量(N 是词表大小),这个向量只有第 i 个元
# 素是 1,其余元素都为 0。
# 当然,也可以进行字符级的 one-hot 编码。

# 单词级的 one-hot 编码
import numpy as np

# 初始数据:每个样本是列表的一个元素(本例中的样本是一个句子,但也可以是一整篇文档)
samples = ['The cat sat on the mat.', 'The dog ate my homework.']

# 构建数据中所有标记的索引
token_index = {}
for sample in samples:
    # 利用 split 方法对样本进行分词。在实际应用中,还需要从样本中去掉标点和特殊字符
    for word in sample.split():
        if word not in token_index:
            # 为每个唯一单词指定一个唯一索引。注意,没有为索引编号 0 指定单词
            token_index[word] = len(token_index) + 1

# 对样本进行分词。只考虑每个样本前 max_length 个单词
max_length = 10

# 将结果保存在 results 中
results = np.zeros((len(samples), max_length, max(token_index.values()) + 1))
for i, sample in enumerate(samples):
    for j, word in list(enumerate(sample.split()))[:max_length]:
        index = token_index.get(word)
        results[i, j, index] = 1.


# 字符级的 one-hot 编码

# In[5]:


import string

samples = ['The cat sat on the mat.', 'The dog ate my homework.']
characters = string.printable  # 所有可打印的 ASCII 字符
token_index = dict(zip(characters, range(1, len(characters) + 1)))

max_length = 50
results = np.zeros((len(samples), max_length, max(token_index.values()) + 1))
for i, sample in enumerate(samples):
    for j, character in enumerate(sample[:max_length]):
        index = token_index.get(character)
        results[i, j, index] = 1.


# 注意,Keras 的内置函数可以对原始文本数据进行单词级或字符级的 one-hot 编码。你应该
# 使用这些函数,因为它们实现了许多重要的特性,比如从字符串中去除特殊字符、只考虑数据
# 集中前 N 个最常见的单词(这是一种常用的限制,以避免处理非常大的输入向量空间)。
# 用 Keras 实现单词级的 one-hot 编码

from keras.preprocessing.text import Tokenizer

samples = ['The cat sat on the mat.', 'The dog ate my homework.']

# 创建一个分词器(tokenizer),设置为只考虑前 1000 个最常见的单词
tokenizer = Tokenizer(num_words=1000)
# 构建单词索引
tokenizer.fit_on_texts(samples)

# 将字符串转换为整数索引组成的列表
sequences = tokenizer.texts_to_sequences(samples)

# 也可以直接得到 one-hot 二进制表示。这个分词器也支持除 one-hot 编码外的其他向量化模式
one_hot_results = tokenizer.texts_to_matrix(samples, mode='binary')

# 找回单词索引
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))


# one-hot 编码的一种变体是所谓的 one-hot 散列技巧(one-hot hashing trick),如果词表中唯
# 一标记的数量太大而无法直接处理,就可以使用这种技巧。这种方法没有为每个单词显式分配
# 一个索引并将这些索引保存在一个字典中,而是将单词散列编码为固定长度的向量,通常用一
# 个非常简单的散列函数来实现。这种方法的主要优点在于,它避免了维护一个显式的单词索引,
# 从而节省内存并允许数据的在线编码(在读取完所有数据之前,你就可以立刻生成标记向量)。
# 这种方法有一个缺点,就是可能会出现散列冲突(hash collision)
# ,即两个不同的单词可能具有
# 相同的散列值,随后任何机器学习模型观察这些散列值,都无法区分它们所对应的单词。如果
# 散列空间的维度远大于需要散列的唯一标记的个数,散列冲突的可能性会减小。



# 使用散列技巧的单词级的 one-hot 编码
samples = ['The cat sat on the mat.', 'The dog ate my homework.']

# 将单词保存为长度为 1000 的向量。如果单词数量接近 1000 个(或更多),那么会遇到很多散列冲突,这会降低这种编码方法的准确性
dimensionality = 1000
max_length = 10

results = np.zeros((len(samples), max_length, dimensionality))
for i, sample in enumerate(samples):
    for j, word in list(enumerate(sample.split()))[:max_length]:
        # 将单词散列为 0~1000 范围内的一个随机整数索引
        index = abs(hash(word)) % dimensionality
        results[i, j, index] = 1.

