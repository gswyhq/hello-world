#!/usr/bin/python3
# coding: utf-8


import numpy as np  #A
import matplotlib.pyplot as plt
import tensorflow as tf

def test():
    """可视化原始输入"""
    x_train = np.linspace(-1, 1, 101)  # 输入值为 -1 到 1 之间的 101 个均匀间隔的数字
    y_train = 2 * x_train + np.random.randn(*x_train.shape) * 0.33  # 生成输出值，与输入值成正比并附加噪声;散点图 y=x+ε，ε 为噪声。

    plt.scatter(x_train, y_train)  # 使用 matplotlib 的函数绘制散点图
    plt.show()


def main():
    """求解线性回归"""

    # 定义学习算法使用的一些常数，称为超参数
    learning_rate = 0.01
    training_epochs = 100

    # 初始化线性模拟数据
    x_train = np.linspace(-1, 1, 101)
    y_train = 2 * x_train + np.random.randn(*x_train.shape) * 0.33

    # 将输入和输出节点设置为占位符，而真实数值将传入 x_train 和 y_train
    X = tf.placeholder("float")
    Y = tf.placeholder("float")

    w = tf.Variable(0.0, name="weights")  # 设置权重变量

    # 定义成本函数
    def model(X, w):
        """将模型定义为 y=w*x"""
        return tf.multiply(X, w)

    y_model = model(X, w)
    cost = tf.square(Y-y_model)

    # 定义在学习算法的每次迭代中将被调用的操作
    train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    # 设置会话并初始化所有变量
    sess = tf.Session()
    init = tf.global_variables_initializer()

    sess.run(init)

    # 多次循环遍历数据集
    for epoch in range(training_epochs):
        #循环遍历数据集中的每个数据
        for (x, y) in zip(x_train, y_train):
            sess.run(train_op, feed_dict={X: x, Y: y})
    # 更新模型参数以尝试最小化成本函数

    # 得到最终参数值
    w_val = sess.run(w)

    # 关闭会话
    sess.close()

    # 绘制原始数据
    plt.scatter(x_train, y_train)

    # 绘制最佳拟合直线
    y_learned = x_train*w_val
    plt.plot(x_train, y_learned, 'r')
    plt.show()






if __name__ == '__main__':
    # test()
    main()