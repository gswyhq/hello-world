#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.python.platform import gfile

from tensorflow import keras
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix

'''读取数据'''
vocab_size = 20000  # Only consider the top 20k words
maxlen = 200  # Only consider the first 200 words of each movie review
(x_train, y_train), (x_val, y_val) = keras.datasets.imdb.load_data(num_words=vocab_size)
print(len(x_train), "Training sequences")
print(len(x_val), "Validation sequences")
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_val = keras.preprocessing.sequence.pad_sequences(x_val, maxlen=maxlen)

def load_pb(pb_file_path):
    # sess = tf.Session()
    sess = tf.compat.v1.Session()
    with gfile.FastGFile(pb_file_path, 'rb') as f:
        # graph_def = tf.GraphDef()
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')

        # 查看模型有哪些层
        [print(n.name) for n in tf.compat.v1.get_default_graph().as_graph_def().node]

    # print(sess.run('Input:0'))
    # 输入
    input_x = sess.graph.get_tensor_by_name('Input:0')
    # 输出
    op = sess.graph.get_tensor_by_name('Identity:0')
    # 预测结果
    ret = sess.run(op, {input_x: x_val[:100]})
    print(confusion_matrix(y_val[:100], ret.argmax(1)))

def load_pb_tf1_14(pb_file_path):
    with tf.Graph().as_default():  # 一定要添加该句，否则预测结果是随机的，压根就不是用训练的模型预测的；
        with open(pb_file_path, 'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
    #         sess.graph.as_default()
            tf.import_graph_def(graph_def, name='')

        # sess = tf.Session()
    #     sess = tf.compat.v1.Session()
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
                # 查看模型有哪些层
        #         [print(n.name) for n in tf.compat.v1.get_default_graph().as_graph_def().node]
    #             sess.run(tf.global_variables_initializer())
            # print(sess.run('Input:0'))
            # 输入
            input_x = sess.graph.get_tensor_by_name('input:0')
            # 输出
            op = sess.graph.get_tensor_by_name('output/Softmax:0')
            # 预测结果
            ret = sess.run(op, {input_x: x_test[:100]})
    #             print(ret)
            print(confusion_matrix(y_test[1][:100], ret.argmax(1)))
load_pb_tf1_14('./result/202206130857/imdb4.pb')

def main():
    load_pb(r"imdb.pb")

if __name__ == '__main__':
    main()