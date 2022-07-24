#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import math
from keras.layers import Input, Lambda, Dense
from keras.models import Model, load_model
import keras.backend as K
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
from tensorflow.keras.utils import Sequence
from keras.callbacks import History

# keras.__version__
# Out[29]: '2.6.0'
# tf.__version__
# Out[30]: '2.6.0'
# keras_bert.__version__
# Out[31]: '0.89.0'

config_path = r'D:\Users\{}\data\RoBERTa-tiny-clue\bert_config.json'.format(os.getenv("USERNAME"))
checkpoint_path = r'D:\Users\{}\data\RoBERTa-tiny-clue\bert_model.ckpt'.format(os.getenv("USERNAME"))
dict_path = r'D:\Users\{}\data\RoBERTa-tiny-clue\vocab.txt'.format(os.getenv("USERNAME"))


token_dict = {}
with open(dict_path, 'r', encoding='utf-8') as f:
    for line in f.readlines():
        token = line.strip()
        token_dict[token] = len(token_dict)

tokenizer = Tokenizer(token_dict)
cut_words = tokenizer.tokenize(u'今天天气不错')
print(cut_words)

def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])


class data_generator:
    def __init__(self, data, batch_size=24):
        self.data = data
        self.batch_size = batch_size
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1
    def __len__(self):
        return self.steps
    def __iter__(self):
        while True:
            idxs = list(range(len(self.data)))
            np.random.shuffle(idxs)
            X1, X2, Y = [], [], []
            for i in idxs:
                d = self.data[i]
                text = d[0][:maxlen]
                x1, x2 = tokenizer.encode(first=text)
                y = d[1]
                X1.append(x1)
                X2.append(x2)
                Y.append([y])
                if len(X1) == self.batch_size or i == idxs[-1]:
                    X1 = seq_padding(X1)
                    X2 = seq_padding(X2)
                    Y = seq_padding(Y)
                    yield [X1, X2], Y
                    [X1, X2, Y] = [], [], []

bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=None)
for l in bert_model.layers:
    l.trainable = True
x1_in = Input(shape=(None,))
x2_in = Input(shape=(None,))
x = bert_model([x1_in, x2_in])
x = Lambda(lambda x: x[:, 0])(x)
p = Dense(1, activation='sigmoid')(x)
model = Model([x1_in, x2_in], p)
model.compile(
    loss='binary_crossentropy',
    optimizer='adam', # 用足够小的学习率
    metrics=['accuracy']
)
model.summary()

def get_features(text_list):
    '''获取文本特征'''
    vec_list = []
    for text in text_list:
        indices, segments = tokenizer.encode(text, second=None, max_len=32)
        predicts = bert_model.predict([np.array([indices]), np.array([segments])])[0]
        vec = predicts[0].tolist()
        vec_list.append(vec)
    return vec_list

class DataGenerator(Sequence):

    def __init__(self, datas, batch_size=256, shuffle=True):
        self.batch_size = batch_size
        self.datas = datas
        self.indexes = np.arange(len(self.datas))
        self.shuffle = shuffle

    def __len__(self):
        # 计算每一个epoch的迭代次数
        return math.ceil(len(self.datas) / float(self.batch_size))

    def __getitem__(self, index):
        # 生成每个batch数据，这里就根据自己对数据的读取方式进行发挥了
        # 生成batch_size个索引
        batch_indexs = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        # 根据索引获取datas集合中的数据
        batch_datas = [self.datas[k] for k in batch_indexs]

        # 生成数据
        X, y = self.data_generation(batch_datas)

        return X, y

    def on_epoch_end(self):
        # 在每一次epoch结束是否需要进行一次随机，重新随机一下index
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def data_generation(self, batch_datas):
        text_a_vec_list = []
        text_b_vec_list = []
        labels = []

        # 生成数据
        for text_a, label in batch_datas:
            # x_train数据
            indices, segments = tokenizer.encode(text_a, second=None, max_len=32)
            # predicts = bert_model.predict([np.array([indices]), np.array([segments])])[0]
            text_a_vec_list.append(indices)
            text_b_vec_list.append(segments)
            labels.append(label)

        return [np.array(text_a_vec_list), np.array(text_b_vec_list)], np.array(labels)


# 数据生成器
train_datas = [[' 也没多大，科研项目', 1],
 ['当时面试没怎么问技术问题,更多的是问个人能力相关的', 0],
 ['唉，深受其苦', 1],
 ['因此不推荐朋友来这里工作', 0],
 ['一技术笔试：HTML，CSS，JS，PHP，问题很基础', 0],
 [' 没有 你从深圳前海高信达通讯有限公司离职的原因是什么', 0],
 ['整体流程比较长，其他还好', 0],
 [' 制度完善，福利不错', 1],
 ['觉得适合应届女孩子嘛', 0],
 ['皇亲国戚比较多', 1],
 ['感觉在这里做销售挣钱是挣不多的', 0],
 ['设计营销大于设计本体的设计公司', 1],
 ['创业型公司，工作压力较大', 0],
 ['现在您还在公司吗', 0],
 ['行政本职应该是为员工服务，却都是坑你，唉', 1],
 ['笔试加上试讲试讲一道题，内容自定', 0],
 ['仑头村 黄埔古港', 1],
 [' 流动大，不适合长期发展', 1],
 [' 为什么支持法人代表 非常有远见，很会用人，谋略清晰', 1],
 ['说是做科研客户的业务，销售部门', 0],
 ['说的谁，带个姓', 1],
 ['还行，但是可能不适合我', 1],
 ['且发布出来的文件没公章，没主题，我们基层员工也看不懂', 0],
 ['这公司难怪一直被人骂', 0],
 ['可以再详细的说下公司的晋升机制嘛', 0],
 [' 发工资准时，晋升级别完整', 0]]
train_generator = DataGenerator(train_datas, batch_size=4)

hist = model.fit_generator(train_generator)
print(hist.history)

# 将对应的h5(hdf5)模型转换为pb模型
def tf2_hdf5_to_pb(h5_save_path, custom_objects=None):
    pb_model_path = os.path.splitext(h5_save_path)[0] + '.pb'
    print(pb_model_path)
    if custom_objects is None:
        custom_objects = {}
    model = tf.keras.models.load_model(h5_save_path, compile=False,
                                       custom_objects=custom_objects  # custom_objects, 参数设置自定义网络层；
                                       )
    model.summary()
    full_model = tf.function(lambda *x: model(x))
    x_tensor_spec = [tf.TensorSpec(shape=input_x.shape, dtype=input_x.dtype, name=input_x.name) for input_x in model.inputs]
    full_model = full_model.get_concrete_function(*x_tensor_spec)

    # Get frozen ConcreteFunction
    frozen_func = convert_variables_to_constants_v2(full_model)
    frozen_func.graph.as_graph_def()

    layers = [op.name for op in frozen_func.graph.get_operations()]
#     print("-" * 50)
#     print("Frozen model layers: ")
#     for layer in layers:
#         print(layer)

#     print("-" * 50)
    print("Frozen model inputs: ")
    print(frozen_func.inputs)
    print("Frozen model outputs: ")
    print(frozen_func.outputs)
    print('input is :', [t.name for t in model.inputs])
    print ('output is:', [t.name for t in model.outputs])

    # Save frozen graph from frozen ConcreteFunction to hard drive
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                      logdir=".",
                      name=pb_model_path,
                      as_text=False)   #可设置.pb存储路径

keras.backend.clear_session()
tf2_hdf5_to_pb('./result/20220626111134/classify-08-0.3903-0.8203.hdf5', custom_objects = {layer.__class__.__name__:layer for layer in bert_model.layers})

def main():
    pass


if __name__ == '__main__':
    main()