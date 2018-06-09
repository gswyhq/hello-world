#!/usr/bin/python3
# coding: utf-8

# 来源： http://www.jianshu.com/p/e112012a4b2d

# 搭建神经网络基本流程
#
# 定义添加神经层的函数
#
# 1.训练的数据
# 2.定义节点准备接收数据
# 3.定义神经层：隐藏层和预测层
# 4.定义 loss 表达式
# 5.选择 optimizer 使 loss 达到最小
#
# 然后对所有变量进行初始化，通过 sess.run optimizer，迭代 1000 次进行学习：


import tensorflow as tf
import numpy as np

IS_TENSORBOARD = True  # 是否需要可视化

# 添加层
def add_layer(inputs, in_size, out_size, activation_function=None):

    if IS_TENSORBOARD:
        # Tensorflow 自带 tensorboard ，可以自动显示我们所建造的神经网络流程图
        # 用 with tf.name_scope 定义各个框架
        with tf.name_scope('layer'):
            with tf.name_scope('weights'):
                Weights = tf.Variable(tf.random_normal([in_size, out_size]))
            with tf.name_scope('biases'):
                biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
            with tf.name_scope('Wx_plus_b'):
                Wx_plus_b = tf.matmul(inputs, Weights) + biases

                # 防止过拟合
                # 在 Wx_plus_b 上drop掉一定比例
                # keep_prob 保持多少不被drop，在迭代时在 sess.run 中 feed
                keep_prob = 0.5  # keep_prob是保留概率，即我们要保留的RELU的结果所占比例
                Wx_plus_b = tf.nn.dropout(Wx_plus_b, keep_prob)
    else:
        # add one more layer and return the output of this layer
        Weights = tf.Variable(tf.random_normal([in_size, out_size]))
        biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
        Wx_plus_b = tf.matmul(inputs, Weights) + biases

        # 防止过拟合
        # 在 Wx_plus_b 上drop掉一定比例
        # keep_prob 保持多少不被drop，在迭代时在 sess.run 中 feed
        keep_prob = 0.5  # keep_prob是保留概率，即我们要保留的RELU的结果所占比例
        Wx_plus_b = tf.nn.dropout(Wx_plus_b, keep_prob)

    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs




# 1.训练的数据
# 随机定义训练的数据 x 和 y：
x_data = np.linspace(-1,1,300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

# 2.定义节点准备接收数据
# define placeholder for inputs to network


if IS_TENSORBOARD:
    with tf.name_scope('inputs'):
        xs = tf.placeholder(tf.float32, [None, 1])
        ys = tf.placeholder(tf.float32, [None, 1])
else:
    xs = tf.placeholder(tf.float32, [None, 1])
    ys = tf.placeholder(tf.float32, [None, 1])

# 3.定义神经层：隐藏层和预测层
# add hidden layer 输入值是 xs，在隐藏层有 10 个神经元
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
# add output layer 输入值是隐藏层 l1，在预测层输出 1 个结果
prediction = add_layer(l1, 10, 1, activation_function=None)

# 4.定义 loss 表达式
# the error between prediciton and real data
if IS_TENSORBOARD:
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                                        reduction_indices=[1]))
else:
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                         reduction_indices=[1]))

# 5.选择 optimizer 使 loss 达到最小
# 这一行定义了用什么方式去减少 loss，学习率是 0.1
if IS_TENSORBOARD:
    with tf.name_scope('train'):
        train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
else:
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# important step 对所有变量进行初始化
init = tf.global_variables_initializer()
sess = tf.Session()
# 上面定义的都没有运算，直到 sess.run 才会开始运算

if IS_TENSORBOARD:
    # 把所有框架加载到一个文件中放到文件夹"logs/"里
    # 接着打开terminal，进入你存放的文件夹地址上一层，运行命令 tensorboard --logdir='logs/'
    # 会返回一个地址，然后用浏览器打开这个地址，在 graph 标签栏下打开
    writer = tf.summary.FileWriter("logs/", sess.graph)

    # 运行完上面代码后，打开 终端，进入你存放的文件夹地址上一层，运行命令 tensorboard --logdir='logs/' 后会返回一个地址，
    # gswewf@gswewf-pc:~/hello-world/TensorFlow$ tensorboard --logdir='logs/'
    # 然后用浏览器打开这个地址，点击 graph 标签栏下就可以看到流程图了

sess.run(init)

# 迭代 1000 次学习，sess.run optimizer
for i in range(1000):
    # training train_step 和 loss 都是由 placeholder 定义的运算，所以这里要用 feed 传入参数
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i % 50 == 0:
        # to see the step improvement
        print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))


def main():
    pass


if __name__ == '__main__':
    main()