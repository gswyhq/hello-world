#!/usr/bin/python3
# coding: utf-8

import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

model_path = './model2'


def train(x_train, y_train):
    """训练模型"""
    x = tf.placeholder(tf.float32, [None, 1], name='x')  # 保存要输入的格式
    w = tf.Variable(tf.zeros([1, 1]))
    b = tf.Variable(tf.zeros([1]))
    y = tf.matmul(x, w) + b
    tf.add_to_collection('pred_network', y)  # 用于加载模型获取要预测的网络结构
    y_ = tf.placeholder(tf.float32, [None, 1])
    cost = tf.reduce_sum(tf.pow((y - y_), 2))
    train_step = tf.train.GradientDescentOptimizer(0.000001).minimize(cost)
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    cost_history = []
    saver = tf.train.Saver()
    for i in range(100):
        feed = {x: x_train, y_: y_train}
        sess.run(train_step, feed_dict=feed)
        cost_history.append(sess.run(cost, feed_dict=feed))

        print("W_Value: %f" % sess.run(w), "b_Value: %f" % sess.run(b),
              "cost_Value: %f" % sess.run(cost, feed_dict=feed))

    # 输出最终的W,b和cost值
    feed = {x: [[109]]}
    print("109的预测值是:", sess.run(y, feed_dict=feed))
    saver_path = saver.save(sess, os.path.join(model_path, "model.ckpt"), global_step=100)
    # 在saver.save的时候，每个checkpoint会保存三个文件，如
    # model.ckpt-100.meta, model.ckpt-100.index, model.ckpt-100.data-00000-of-00001
    print("model saved in file: ", saver_path)

class RestoreModel():
    """加载训练的模型并进行预测"""
    def __init__(self):
        checkpoint_file = tf.train.latest_checkpoint(model_path)
        graph = tf.Graph()
        with graph.as_default():
            self.sess = tf.Session()
            # 在import_meta_graph时填的就是meta文件名，我们知道权值都保存在model.ckpt-100.data-00000-of-00001这个文件中，
            # 但是如果在restore方法中填这个文件名，就会报错，应该填的是前缀，
            # 这个前缀可以使用tf.train.latest_checkpoint(checkpoint_dir)这个方法获取。

            new_saver = tf.train.import_meta_graph('{}.meta'.format(checkpoint_file))
            new_saver.restore(self.sess, checkpoint_file)
            self.x = graph.get_operation_by_name('x').outputs[0]
            self.y = tf.get_collection("pred_network")[0]

    def predict(self, x_test, y_test):
            print("{}的预测值是:".format(x_test), self.sess.run(self.y, feed_dict={self.x: x_test}))
            print('实际值：{}'.format(y_test))



def main():
    money = np.array([[109], [82], [99], [72], [87], [78], [86], [84], [94], [57]]).astype(np.float32)
    click = np.array([[11], [8], [8], [6], [7], [7], [7], [8], [9], [5]]).astype(np.float32)
    x_test = money[0:5].reshape(-1, 1)
    y_test = click[0:5]
    x_train = money[5:].reshape(-1, 1)
    y_train = click[5:]
    # 训练一个线性回归模型并保存模型
    # train(x_train, y_train)

    model = RestoreModel()
    model.predict(x_test, y_test)

if __name__ == '__main__':
    main()