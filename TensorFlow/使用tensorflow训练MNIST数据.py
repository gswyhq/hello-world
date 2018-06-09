#!/usr/bin/python3
# coding: utf-8

import tensorflow as tf
import numpy as np

# 来源： http://blog.topspeedsnail.com/archives/10377

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

# 定义每个层有多少'神经元''
n_input_layer = 28 * 28  # 输入层

n_layer_1 = 500  # hide layer
n_layer_2 = 1000  # hide layer
n_layer_3 = 300  # hide layer(隐藏层)听着很神秘，其实就是除输入输出层外的中间层

n_output_layer = 10  # 输出层
"""
层数的选择：线性数据使用1层，非线性数据使用2册, 超级非线性使用3+册。层数／神经元过多会导致过拟合
"""


# 定义待训练的神经网络(feedforward)
def neural_network(data):
    # 定义第一层"神经元"的权重和biases
    layer_1_w_b = {'w_': tf.Variable(tf.random_normal([n_input_layer, n_layer_1])),
                   'b_': tf.Variable(tf.random_normal([n_layer_1]))}
    # 定义第二层"神经元"的权重和biases
    layer_2_w_b = {'w_': tf.Variable(tf.random_normal([n_layer_1, n_layer_2])),
                   'b_': tf.Variable(tf.random_normal([n_layer_2]))}
    # 定义第三层"神经元"的权重和biases
    layer_3_w_b = {'w_': tf.Variable(tf.random_normal([n_layer_2, n_layer_3])),
                   'b_': tf.Variable(tf.random_normal([n_layer_3]))}
    # 定义输出层"神经元"的权重和biases
    layer_output_w_b = {'w_': tf.Variable(tf.random_normal([n_layer_3, n_output_layer])),
                        'b_': tf.Variable(tf.random_normal([n_output_layer]))}

    # w·x+b
    layer_1 = tf.add(tf.matmul(data, layer_1_w_b['w_']), layer_1_w_b['b_'])
    layer_1 = tf.nn.relu(layer_1)  # 激活函数
    layer_2 = tf.add(tf.matmul(layer_1, layer_2_w_b['w_']), layer_2_w_b['b_'])
    layer_2 = tf.nn.relu(layer_2)  # 激活函数
    layer_3 = tf.add(tf.matmul(layer_2, layer_3_w_b['w_']), layer_3_w_b['b_'])
    layer_3 = tf.nn.relu(layer_3)  # 激活函数
    layer_output = tf.add(tf.matmul(layer_3, layer_output_w_b['w_']), layer_output_w_b['b_'])

    return layer_output


# 每次使用100条数据进行训练
batch_size = 100

X = tf.placeholder('float', [None, 28 * 28])
# [None, 28*28]代表数据数据的高和宽（矩阵），好处是如果数据不符合宽高，tensorflow会报错，不指定也可以。
Y = tf.placeholder('float')


# 使用数据训练神经网络
def train_neural_network(X, Y):
    predict = neural_network(X)
    cost_func = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(predict, Y))
    optimizer = tf.train.AdamOptimizer().minimize(cost_func)  # learning rate 默认 0.001

    epochs = 13
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        epoch_loss = 0
        for epoch in range(epochs):
            for i in range(int(mnist.train.num_examples / batch_size)):
                x, y = mnist.train.next_batch(batch_size)
                _, c = session.run([optimizer, cost_func], feed_dict={X: x, Y: y})
                epoch_loss += c
            print(epoch, ' : ', epoch_loss)

        # print(predict.eval(feed_dict={X:[features]}))
        correct = tf.equal(tf.argmax(predict, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('准确率: ', accuracy.eval({X: mnist.test.images, Y: mnist.test.labels}))



def main():
    train_neural_network(X, Y)


if __name__ == '__main__':
    main()