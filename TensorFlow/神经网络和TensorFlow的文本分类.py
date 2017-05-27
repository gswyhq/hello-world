#!/usr/bin/python3
# coding: utf-8

# tensors 是通过图的节点流转的多维数组。
# 在 TensorFlow 中的每一个计算都表示为数据流图，这个图有两类元素：
# 一类 tf.Operation，表示计算单元
# 一类 tf.Tensor，表示数据单元

# TensorFlow 工作流的工作原理：首先要创建一个图，然后你才能计算（实际上是用操作‘运行’图节点）。
# 你需要创建一个 tf.Session 运行图。

import pandas as pd
import numpy as np
import tensorflow as tf
from collections import Counter
from sklearn.datasets import fetch_20newsgroups
my_graph = tf.Graph()  # 构建图

# tf.Session 对象封装了 Operation 对象的执行环境。Tensor 对象是被计算过的（从文档中）。
# 为了做到这些，我们需要在 Session 中定义哪个图将被使用到：
with tf.Session(graph=my_graph) as sess:

    # 由于图用 tf.Tensor 表示数据单元;
    # 需要定义 x = [1,3,6] 和 y = [1,1,1]。由于图用 tf.Tensor 表示数据单元，你需要创建常量 Tensors
    x = tf.constant([1,3,6])
    y = tf.constant([1,1,1])
    op = tf.add(x,y)  # 定义操作单元
    result = sess.run(fetches=op)  # 为了执行操作，你需要使用方法 tf.Session.run()。这个方法通过运行必要的图段去执行每个 Operation 对象;
    # 并通过参数 fetches 计算每一个 Tensor 的值的方式执行 TensorFlow 计算的一'步'
    print(result)

vocab = Counter()

text = "Hi from Brazil"

for word in text.split(' '):
    word_lowercase = word.lower()
    vocab[word_lowercase] += 1


def get_word_2_index(vocab):
    word2index = {}
    for i, word in enumerate(vocab):
        word2index[word] = i

    return word2index


word2index = get_word_2_index(vocab)

total_words = len(vocab)
matrix = np.zeros((total_words), dtype=float)

for word in text.split():
    matrix[word2index[word.lower()]] += 1

print("Hi from Brazil:", matrix)

matrix = np.zeros((total_words), dtype=float)
text = "Hi"
for word in text.split():
    matrix[word2index[word.lower()]] += 1

print("Hi:", matrix)
# 对于一个有 18.000 个帖子大约有 20 个主题的数据集，将会使用到 20个新闻组。要加载这些数据集将会用到 scikit-learn 库。我们只使用 3 种类别：
# comp.graphics
# sci.space
# rec.sport.baseball
# scikit-learn 有两个子集：一个用于训练，另一个用于测试。
categories = ["comp.graphics","sci.space","rec.sport.baseball"]
newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)
newsgroups_test = fetch_20newsgroups(subset='test', categories=categories)

print('total texts in train:',len(newsgroups_train.data))  # 1774
print('total texts in test:',len(newsgroups_test.data))  # 1180

print('text',newsgroups_train.data[0])
print('category:',newsgroups_train.target[0])  # 0

vocab = Counter()

for text in newsgroups_train.data:
    for word in text.split(' '):
        vocab[word.lower()] += 1

for text in newsgroups_test.data:
    for word in text.split(' '):
        vocab[word.lower()] += 1

print("Total words:",len(vocab))  # 119930

total_words = len(vocab)


def get_word_2_index(vocab):
    word2index = {}
    for i, word in enumerate(vocab):
        word2index[word.lower()] = i

    return word2index


word2index = get_word_2_index(vocab)

print("Index of the word 'the':", word2index['the'])  # 116011

# 提供批处理大小的文本数
def get_batch(df, i, batch_size):
    batches = []
    results = []
    texts = df.data[i * batch_size:i * batch_size + batch_size]
    categories = df.target[i * batch_size:i * batch_size + batch_size]
    for text in texts:
        layer = np.zeros(total_words, dtype=float)
        for word in text.split(' '):
            layer[word2index[word.lower()]] += 1

        batches.append(layer)

    for category in categories:
        # 对应三个分类：sports, space 和computer graphics
        y = np.zeros((3), dtype=float)
        if category == 0:
            y[0] = 1.
        elif category == 1:
            y[1] = 1.
        else:
            y[2] = 1.
        results.append(y)

    return np.array(batches), np.array(results)

print("Each batch has 100 texts and each matrix has 119930 elements (words):",get_batch(newsgroups_train,1,100)[0].shape)
print("Each batch has 100 labels and each matrix has 3 elements (3 categories):",get_batch(newsgroups_train,1,100)[1].shape)

# 参数
learning_rate = 0.01
training_epochs = 10
batch_size = 150
display_step = 1

# Network Parameters
n_hidden_1 = 100      # 第一层特征数
n_hidden_2 = 100       # 第二层特征数
n_input = total_words # Words in vocab
n_classes = 3         # 三种分类: graphics, sci.space and baseball

# 需要定义tf.placeholders（提供给 feed_dict）
input_tensor = tf.placeholder(tf.float32,[None, n_input],name="input")
output_tensor = tf.placeholder(tf.float32,[None, n_classes],name="output")


def multilayer_perceptron(input_tensor, weights, biases):
    layer_1_multiplication = tf.matmul(input_tensor, weights['h1'])
    layer_1_addition = tf.add(layer_1_multiplication, biases['b1'])
    layer_1 = tf.nn.relu(layer_1_addition)

    # 隐藏层 RELU 激活函数
    layer_2_multiplication = tf.matmul(layer_1, weights['h2'])
    layer_2_addition = tf.add(layer_2_multiplication, biases['b2'])
    layer_2 = tf.nn.relu(layer_2_addition)

    # 输出层
    out_layer_multiplication = tf.matmul(layer_2, weights['out'])
    out_layer_addition = out_layer_multiplication + biases['out']

    return out_layer_addition

# 权重和误差存储在变量（tf.Variable）中。这些变量通过调用 run() 保持在图中的状态。
# 在机器学习中我们一般通过 正太分布 来启动权重和偏差值。
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# 构建 model
prediction = multilayer_perceptron(input_tensor, weights, biases)

# 定义 loss and optimizer
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=output_tensor))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# Initializing the variables
init = tf.global_variables_initializer()

# 运行图
with tf.Session() as sess:
    sess.run(init)

    # 训练周期
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(len(newsgroups_train.data)/batch_size)
        # 循环所有批次
        for i in range(total_batch):
            batch_x,batch_y = get_batch(newsgroups_train,i,batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            # feed_dict 参数是我们为每步运行所输入的数据。
            c,_ = sess.run([loss,optimizer], feed_dict={input_tensor: batch_x,output_tensor:batch_y})
            # 计算平均损失
            avg_cost += c / total_batch
        # 打印每一步的效果
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "loss=", \
                "{:.9f}".format(avg_cost))
    print("Optimization Finished!")

    # 测试 model
    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(output_tensor, 1))
    # 计算准确度
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    total_test_data = len(newsgroups_test.target)
    batch_x_test,batch_y_test = get_batch(newsgroups_test,0,total_test_data)
    print("准确度:", accuracy.eval({input_tensor: batch_x_test, output_tensor: batch_y_test}))


def main():
    pass


if __name__ == '__main__':
    main()