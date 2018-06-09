#!/usr/bin/python3
# coding: utf-8

import tensorflow as tf
import numpy as np

# 来源： http://blog.topspeedsnail.com/archives/10443

"""
与Feed-forward Neural Networks(FNN/前向反馈神经网络) 相比，RNN(Recurrent Neural Networks)循环神经网络要相对复杂，它引入了循环，
能够处理数据有前后关系的问题。
RNN的目的使用来处理序列数据。在传统的神经网络模型中，是从输入层到隐含层再到输出层，层与层之间是全连接的，每层之间的节点是无连接的。
但是这种普通的神经网络对于很多问题却无能无力。例如，你要预测句子的下一个单词是什么，一般需要用到前面的单词，因为一个句子中前后单词并不是独立的。
RNNs之所以称为循环神经网路，即一个序列当前的输出与前面的输出也有关。具体的表现形式为网络会对前面的信息进行记忆并应用于当前输出的计算中，
即隐藏层之间的节点不再无连接而是有连接的，并且隐藏层的输入不仅包括输入层的输出还包括上一时刻隐藏层的输出。

"""
# tensorflow自带了MNIST数据集
from tensorflow.examples.tutorials.mnist import input_data

# 下载mnist数据集
# mnist = input_data.read_data_sets('/tmp/', one_hot=True)
mnist = input_data.read_data_sets('/home/gswewf/MNIST_data', one_hot=True)
# 数字(label)只能是0-9，神经网络使用10个出口节点就可以编码表示0-9；
#  1 -> [0,1.0,0,0,0,0,0,0,0]   one_hot表示只有一个出口节点是hot
#  2 -> [0,0.1,0,0,0,0,0,0,0]
#  5 -> [0,0,0,0,0,1.0,0,0,0]
#  /tmp是macOS的临时目录，重启系统数据丢失; Linux的临时目录也是/tmp

# 一张图片是28*28,FNN是一次性把数据输入到网络，RNN把它分成块
chunk_size = 28
chunk_n = 28

rnn_size = 256

n_output_layer = 10  # 输出层

X = tf.placeholder('float', [None, chunk_n, chunk_size])
Y = tf.placeholder('float')


# 定义待训练的神经网络
def recurrent_neural_network(data):
    layer = {'w_': tf.Variable(tf.random_normal([rnn_size, n_output_layer])),
             'b_': tf.Variable(tf.random_normal([n_output_layer]))}

    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_size)

    data = tf.transpose(data, [1, 0, 2])
    data = tf.reshape(data, [-1, chunk_size])
    data = tf.split(0, chunk_n, data)
    outputs, status = tf.nn.rnn(lstm_cell, data, dtype=tf.float32)

    ouput = tf.add(tf.matmul(outputs[-1], layer['w_']), layer['b_'])

    return ouput


# 每次使用100条数据进行训练
batch_size = 100


# 使用数据训练神经网络
def train_neural_network(X, Y):
    predict = recurrent_neural_network(X)
    cost_func = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(predict, Y))
    optimizer = tf.train.AdamOptimizer().minimize(cost_func)

    epochs = 13
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        epoch_loss = 0
        for epoch in range(epochs):
            for i in range(int(mnist.train.num_examples / batch_size)):
                x, y = mnist.train.next_batch(batch_size)
                x = x.reshape([batch_size, chunk_n, chunk_size])
                _, c = session.run([optimizer, cost_func], feed_dict={X: x, Y: y})
                epoch_loss += c
            print(epoch, ' : ', epoch_loss)

        correct = tf.equal(tf.argmax(predict, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('准确率: ', accuracy.eval({X: mnist.test.images.reshape(-1, chunk_n, chunk_size), Y: mnist.test.labels}))

# 准确率比FNN高，但训练时间同样要长；

def main():
    train_neural_network(X, Y)


if __name__ == '__main__':
    main()