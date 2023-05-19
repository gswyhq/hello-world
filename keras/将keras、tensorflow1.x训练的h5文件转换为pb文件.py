#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from keras.preprocessing import sequence
from keras.datasets import imdb
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

old = np.load
np.load = lambda *a, **k: old(*a, **k, allow_pickle=True)

max_features = 20000

print('Loading data...')

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

# 标签转换为独热码
y_train, y_test = pd.get_dummies(y_train), pd.get_dummies(y_test)

print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

# %%数据归一化处理

maxlen = 64

print('Pad sequences (samples x time)')

x_train = sequence.pad_sequences(x_train, maxlen=maxlen)

x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

print('x_train shape:', x_train.shape)

print('x_test shape:', x_test.shape)

import tensorflow as tf
import keras
from keras import backend as K

from tensorflow.python.platform import gfile
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io
from keras.engine.topology import Layer

print(keras.__version__, tf.__version__)
# 2.2.4 1.14.0


class Position_Embedding(Layer):

    def __init__(self, size=None, mode='sum', **kwargs):
        self.size = size  # 必须为偶数
        self.mode = mode
        super(Position_Embedding, self).__init__(**kwargs)

    def call(self, x):
        if (self.size == None) or (self.mode == 'sum'):
            self.size = int(x.shape[-1])
        batch_size, seq_len = K.shape(x)[0], K.shape(x)[1]
        position_j = 1. / K.pow(10000., \
                                2 * K.arange(self.size / 2, dtype='float32' \
                                             ) / self.size)
        position_j = K.expand_dims(position_j, 0)
        position_i = K.cumsum(K.ones_like(x[:, :, 0]), 1) - 1  # K.arange不支持变长，只好用这种方法生成
        position_i = K.expand_dims(position_i, 2)
        position_ij = K.dot(position_i, position_j)
        position_ij = K.concatenate([K.cos(position_ij), K.sin(position_ij)], 2)
        if self.mode == 'sum':
            return position_ij + x
        elif self.mode == 'concat':
            return K.concatenate([position_ij, x], 2)

    def compute_output_shape(self, input_shape):
        if self.mode == 'sum':
            return input_shape
        elif self.mode == 'concat':
            return (input_shape[0], input_shape[1], input_shape[2] + self.size)


class Attention(Layer):

    def __init__(self, nb_head=8, size_per_head=16, output_dim=8 * 16, **kwargs):
        self.nb_head = nb_head
        self.size_per_head = size_per_head
        output_dim = nb_head * size_per_head
        self.output_dim = output_dim
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.WQ = self.add_weight(name='WQ',
                                  shape=(input_shape[0][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WK = self.add_weight(name='WK',
                                  shape=(input_shape[1][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WV = self.add_weight(name='WV',
                                  shape=(input_shape[2][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        super(Attention, self).build(input_shape)

    def Mask(self, inputs, seq_len, mode='mul'):
        if seq_len == None:
            return inputs
        else:
            mask = K.one_hot(seq_len[:, 0], K.shape(inputs)[1])
            mask = 1 - K.cumsum(mask, 1)
            for _ in range(len(inputs.shape) - 2):
                mask = K.expand_dims(mask, 2)
            if mode == 'mul':
                return inputs * mask
            if mode == 'add':
                return inputs - (1 - mask) * 1e12

    def call(self, x):
        # 如果只传入Q_seq,K_seq,V_seq，那么就不做Mask
        # 如果同时传入Q_seq,K_seq,V_seq,Q_len,V_len，那么对多余部分做Mask
        if len(x) == 3:
            Q_seq, K_seq, V_seq = x
            Q_len, V_len = None, None
        elif len(x) == 5:
            Q_seq, K_seq, V_seq, Q_len, V_len = x
        # 对Q、K、V做线性变换
        Q_seq = K.dot(Q_seq, self.WQ)
        Q_seq = K.reshape(Q_seq, (-1, K.shape(Q_seq)[1], self.nb_head, self.size_per_head))
        Q_seq = K.permute_dimensions(Q_seq, (0, 2, 1, 3))
        K_seq = K.dot(K_seq, self.WK)
        K_seq = K.reshape(K_seq, (-1, K.shape(K_seq)[1], self.nb_head, self.size_per_head))
        K_seq = K.permute_dimensions(K_seq, (0, 2, 1, 3))
        V_seq = K.dot(V_seq, self.WV)
        V_seq = K.reshape(V_seq, (-1, K.shape(V_seq)[1], self.nb_head, self.size_per_head))
        V_seq = K.permute_dimensions(V_seq, (0, 2, 1, 3))
        # 计算内积，然后mask，然后softmax
        A = K.batch_dot(Q_seq, K_seq, axes=[3, 3]) / self.size_per_head ** 0.5
        A = K.permute_dimensions(A, (0, 3, 2, 1))
        A = self.Mask(A, V_len, 'add')
        A = K.permute_dimensions(A, (0, 3, 2, 1))
        A = K.softmax(A)
        # 输出并mask
        O_seq = K.batch_dot(A, V_seq, axes=[3, 2])
        O_seq = K.permute_dimensions(O_seq, (0, 2, 1, 3))
        O_seq = K.reshape(O_seq, (-1, K.shape(O_seq)[1], self.output_dim))
        O_seq = self.Mask(O_seq, Q_len, 'mul')
        return O_seq

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.output_dim)


# from tensorflow.python.keras.models import load_model   #from keras.models import load_model
from keras.models import load_model
import tensorflow as tf
from keras import backend as K
# from tensorflow.python.keras import backend as K        #from keras import backend as K
from tensorflow.python.framework import graph_io


def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph


"""----------------------------------配置路径-----------------------------------"""
h5_model_path = "./result/202206130857/imdb.h5"  # 填写.h5路径
output_path = "./result/202206130857"
pb_model_name = 'imdb4.pb'  # 填写保存.pb路径

"""----------------------------------导入keras模型------------------------------"""
K.set_learning_phase(0)
net_model = load_model(h5_model_path, compile=False,
                       custom_objects={"Position_Embedding": Position_Embedding, "Attention": Attention})

ret1 = net_model.predict(x_test[:100])
print(confusion_matrix(y_test[1][:100], ret1.argmax(1)))

print('input is :', net_model.input.name)
print('output is:', net_model.output.name)

"""----------------------------------保存为.pb格式------------------------------"""
sess = K.get_session()
frozen_graph = freeze_session(K.get_session(), output_names=[net_model.output.op.name])
graph_io.write_graph(frozen_graph, output_path, pb_model_name, as_text=False)



"""----------------------------------加载.pb格式模型进行预测------------------------------"""
pb_file_path = './result/202206130857/imdb4.pb'
with tf.Graph().as_default():
    output_graph_def = tf.compat.v1.GraphDef()

    # 打开.pb模型
    with open(pb_file_path, "rb") as f:
        output_graph_def.ParseFromString(f.read())
        tensors = tf.import_graph_def(output_graph_def, name="")
        print("tensors:", tensors)

    # 在一个session中去run一个前向
    with tf.compat.v1.Session() as sess:
        init = tf.compat.v1.global_variables_initializer()
        sess.run(init)

        op = sess.graph.get_operations()

        #         net_model = load_model(h5_model_path)
        #         print('input is :', net_model.input.name)
        #         print ('output is:', net_model.output.name)

        input_x = sess.graph.get_tensor_by_name("input:0")  # 具体名称看上一段代码的input.name
        print("input_X:", input_x)

        out_softmax = sess.graph.get_tensor_by_name("output/Softmax:0")  # 具体名称看上一段代码的output.name
        print("Output:", out_softmax)

        ret = sess.run(out_softmax, {input_x: x_test[:100]})
        #             print(ret)
        print(confusion_matrix(y_test[1][:100], ret.argmax(1)))

"""----------------------------------TF1.15 --> TF1.13: 转到TF1.15的pb在TF1.13下会报错("NodeDef mentions attr 'batch_dims' not in Op")------------------------------"""
# 借助SavedModel格式将TF1.15与TF1.13的pb模型打通(这样也打通了TF2与TF1.13),  那么将TF2的pb转为TF1.13的pb具体步骤为:
# 1、在TF2.2中将mode.save()保存的模型转为TF1.15中的pb；
# 2、在TF1.15中读入第一步转好的pb, 并在TF1.15中存为SavedModel格式的模型；
# 3、在TF1.13中读入第二步SavedModel模型, 并在TF1.13中将其存为Frozen Graph格式的pb模型；
# 4、在TF1.13中读入上一步转好的pb, 进行推理预测.

# 在TF1.15中将pb转为SavedModel  # 环境: TF1.15

import os
import tensorflow as tf
from tensorflow.python.saved_model import signature_constants, tag_constants

# 准备输入输出地址
tf115_pb_model = 'path_to_tf1.15_frozen_graph_model.pb'
tf115_saved_model = 'path_to_output_dir'

# 读入TF1.15 pb模型
with tf.gfile.GFile(pb_model, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

# 创建builder, 指定输出路径
builder = tf.saved_model.builder.SavedModelBuilder(tf115_saved_model)

# 通过SavedModelBuild及指定signature输出SavedModelmoxing
sigs = {}
with tf.Session(graph=tf.Graph()) as sess:
    tf.import_graph_def(graph_def, name="")
    seq_wordids = sess.graph.get_tensor_by_name("seq_wordids:0")
    predict_probs = sess.graph.get_tensor_by_name("predict_probs:0")
    sigs[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY] = \
        tf.saved_model.signature_def_utils.predict_signature_def(
            {'seq_wordids': seq_wordids},
            {'predict_probs': predict_probs})
    builder.add_meta_graph_and_variables(sess,
                                         [tag_constants.SERVING],
                                         signature_def_map=sigs)
builder.save()
# 以上输出模型的目录结构为:
# -tf115_save_model
#     -saved_model.pb
#     variables/


# 在TF1.13中读入第二步SavedModel模型, 并在TF1.13中将其存为FrozenGraph格式的pb模型:
# 环境: TF1.13
# tf113_savedmodel_to_pb.py

import tensorflow as tf
from tensorflow.python.framework.graph_util import convert_variables_to_constants

# 准备输入输出路径
tf115_saved_model_dir = 'path_to_tf115_savedmodel_dir'
tf113_pb_model_dir = 'path_to_output_tf113_pb'

# 在tf1.13中读取tf1.15保存的SavedModel
with tf.Session(graph=tf.Graph()) as sess:
    tf.saved_model.loader.load(sess, ['serve'], saved_model_dir)
    graph = tf.get_default_graph()

    # 测试tf1.13可以用读入的模型来推理(可选)
    feed_dict = {"seq_wordids:0": [[0] * 32]}
    x = sess.run('predict_probs:0', feed_dict=feed_dict)
    print(x)

    # 将模型转存为tf1.13的frozen graph的pb格式
    out_graph_def = convert_variables_to_constants(
        sess,
        sess.graph_def,
        output_node_names=['predict_probs'])
    tf.train.write_graph(out_graph_def, tf113_pb_model_dir, 'model.pb', as_text=False)

# 来源：https://zhuanlan.zhihu.com/p/382652354


def main():
    pass


if __name__ == '__main__':
    main()