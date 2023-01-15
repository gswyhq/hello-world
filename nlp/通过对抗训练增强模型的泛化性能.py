#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 通过对抗训练增强模型的泛化性能
# 数据集：IFLYTEK' 长文本分类 (https://github.com/CLUEbenchmark/CLUE)
# 来源：https://kexue.fm/archives/7234
# 来源：https://github.com/bojone/bert4keras/tree/master/examples

'''
深度学习中的对抗，一般会有两个含义：一个是生成对抗网络（Generative Adversarial Networks，GAN），代表着一大类先进的生成模型；另一个则是跟对抗攻击、对抗样本相关的领域，它跟GAN相关，但又很不一样，它主要关心的是模型在小扰动下的稳健性。

对抗样本: 是指对于人类来说“看起来”几乎一样、但对于模型来说预测结果却完全不一样的样本
对抗攻击: 就是想办法造出更多的对抗样本
对抗防御: 就是想办法让模型能正确识别更多的对抗样本
对抗训练: 属于对抗防御的一种，它构造了一些对抗样本加入到原数据集中，希望增强模型对对抗样本的鲁棒性

对抗训练步骤：
1、往属于x里边注入扰动Δx(Δx∈Ω)，Δx的目标是让L(x+Δx,y;θ)越大越好，也就是说尽可能让现有模型的预测出错；
2、当然Δx也不是无约束的，它不能太大，否则达不到“看起来几乎一样”的效果，所以Δx要满足一定的约束，常规的约束是∥Δx∥≤ϵ，其中ϵ是一个常数；
3、每个样本都构造出对抗样本x+Δx之后，用(x+Δx,y)作为数据对去最小化loss来更新参数θ（梯度下降）；
4、反复交替执行1、2、3步。
其中D代表训练集，x代表输入，y代表标签，θ是模型参数，L(x,y;θ)是单个样本的loss，Δx是对抗扰动，Ω是扰动空间。

需要指出的是，由于每一步算对抗扰动也需要计算梯度，因此每一步训练一共算了两次梯度，因此每步的训练时间会翻倍。
将扰动加到Embedding层。
Embedding层的输出是直接取自于Embedding参数矩阵的，因此我们可以直接对Embedding参数矩阵进行扰动。
'''


import os
os.environ['TF_KERAS'] = '1'
USERNAME = os.getenv("USERNAME")
import random
import json, copy
import numpy as np
import tensorflow.keras.backend as K
from tensorflow import keras
from bert4keras.backend import search_layer
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from tensorflow.keras.layers import Lambda, Dense
from keras.engine import data_adapter
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.losses import sparse_categorical_crossentropy, categorical_crossentropy
from tensorflow.python.ops import math_ops
# tf.compat.v1.disable_eager_execution()

num_classes = 119
maxlen = 128
batch_size = 32

# # BERT base
# config_path = rf'D:/Users/{USERNAME}/data/chinese_L-12_H-768_A-12/bert_config.json'
# checkpoint_path = rf'D:/Users/{USERNAME}/data/chinese_L-12_H-768_A-12/bert_model.ckpt'
# dict_path = rf'D:/Users/{USERNAME}/data/chinese_L-12_H-768_A-12/vocab.txt'

# Robert配置
# https://github.com/ZhuiyiTechnology/pretrained-models
# https://open.zhuiyi.ai/releases/nlp/models/zhuiyi/chinese_roberta_L-4_H-312_A-12.zip
config_path = rf'D:/Users/{USERNAME}/data/chinese_roberta_L-4_H-312_A-12/bert_config.json'
checkpoint_path = rf'D:/Users/{USERNAME}/data/chinese_roberta_L-4_H-312_A-12/bert_model.ckpt'
dict_path = rf'D:/Users/{USERNAME}/data/chinese_roberta_L-4_H-312_A-12/vocab.txt'

def load_data(filename):
    """加载数据
    单条格式：(文本, 标签id)
    """
    D = []
    with open(filename, encoding='utf-8') as f:
        for i, l in enumerate(f):
            l = json.loads(l)
            text, label = l['sentence'], l['label']
            D.append((text, int(label)))
    return D


# 加载数据集
train_data = load_data(
    rf'D:/Users/{USERNAME}/data/iflytek/train.json'
)
valid_data = load_data(
    rf'D:/Users/{USERNAME}/data/iflytek/dev.json'
)

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

def data_generator2(train_data, batch_size, shuffle=True):
    while True:
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        if shuffle:
            random.shuffle(train_data)
        for text, label in train_data:
            token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([label])
            if len(batch_token_ids) == batch_size:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield {'Input-Token': batch_token_ids, 'Input-Segment': batch_segment_ids}, batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []

def data_generator3(train_data, batch_size=None, shuffle=True):

    batch_token_ids, batch_segment_ids, batch_labels = [], [], []
    if shuffle:
        random.shuffle(train_data)
    for text, label in train_data:
        token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
        batch_token_ids.append(token_ids)
        batch_segment_ids.append(segment_ids)
        batch_labels.append([label])
    batch_token_ids = sequence_padding(batch_token_ids)
    batch_segment_ids = sequence_padding(batch_segment_ids)
    batch_labels = sequence_padding(batch_labels)
    return {'Input-Token': batch_token_ids, 'Input-Segment': batch_segment_ids}, batch_labels

# train_generator = data_generator2(train_data, batch_size)
# valid_generator = data_generator2(valid_data, batch_size)

# 加载预训练模型
bert = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    return_keras_model=False,
)

output = Lambda(lambda x: x[:, 0])(bert.model.output)
output = Dense(
    units=num_classes,
    activation='softmax',
    kernel_initializer=bert.initializer
)(output)

model = keras.models.Model(bert.model.input, output)
model.summary()
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=Adam(2e-5),
    metrics=['sparse_categorical_accuracy'],
)


def adversarial_training(model, embedding_name, epsilon=1):
    """给模型添加对抗训练
    其中model是需要添加对抗训练的keras模型，embedding_name
    则是model里边Embedding层的名字。要在模型compile之后使用。
    """
    if model.train_function is None:  # 如果还没有训练函数
        model._make_train_function()  # 手动make
    old_train_function = model.train_function  # 备份旧的训练函数

    # 查找Embedding层
    for output in model.outputs:
        embedding_layer = search_layer(output, embedding_name)
        if embedding_layer is not None:
            break
    if embedding_layer is None:
        raise Exception('Embedding layer not found')

    # 求Embedding梯度
    embeddings = embedding_layer.embeddings  # Embedding矩阵
    gradients = K.gradients(model.total_loss, [embeddings])  # Embedding梯度
    gradients = K.zeros_like(embeddings) + gradients[0]  # 转为dense tensor

    # 封装为函数
    inputs = (
        model._feed_inputs + model._feed_targets + model._feed_sample_weights
    )  # 所有输入层
    embedding_gradients = K.function(
        inputs=inputs,
        outputs=[gradients],
        name='embedding_gradients',
    )  # 封装为函数

    def train_function(inputs):  # 重新定义训练函数
        grads = embedding_gradients(inputs)[0]  # Embedding梯度
        delta = epsilon * grads / (np.sqrt((grads**2).sum()) + 1e-8)  # 计算扰动
        K.set_value(embeddings, K.eval(embeddings) + delta)  # 注入扰动
        outputs = old_train_function(inputs)  # 梯度下降
        K.set_value(embeddings, K.eval(embeddings) - delta)  # 删除扰动
        return outputs

    model.train_function = train_function  # 覆盖原训练函数


# 写好函数后，启用对抗训练只需要一行代码
# adversarial_training(model, 'Embedding-Token', 0.5)

def predict_to_file(in_file, out_file):
    """输出预测结果到文件
    结果文件可以提交到 https://www.cluebenchmarks.com 评测。
    """
    fw = open(out_file, 'w', encoding='utf-8')
    with open(in_file, encoding='utf-8') as fr:
        for l in tqdm(fr):
            l = json.loads(l)
            text = l['sentence']
            token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
            label = model.predict([[token_ids], [segment_ids]])[0].argmax()
            l = json.dumps({'id': str(l['id']), 'label': str(label)})
            fw.write(l + '\n')
    fw.close()

class GradientCalcCallback(keras.callbacks.Callback):
    def __init__(self, train_x, train_y, model, batch_size, *args, **kwargs):
        self.last_logs = None
        self.train_x = train_x
        self.train_y = train_y
        self.model = model
        self.batch_size = batch_size
        self.delta = None
        self.epsilon = 0.5
        super(keras.callbacks.Callback, self).__init__(*args, **kwargs)

    def on_batch_begin(self, batch, logs=None):
        # print("begin batch: {}, logs: {}".format(batch, self.last_logs))
        embedding_name = 'Embedding-Token'
        epsilon = self.epsilon
        with tf.GradientTape() as tape:
            # 查找Embedding层
            embeddings = self.model.get_layer(embedding_name).embeddings
            inputs = {'Input-Token': self.train_x['Input-Token'][self.batch_size*batch: self.batch_size*batch+self.batch_size],
                      'Input-Segment': self.train_x['Input-Segment'][self.batch_size*batch: self.batch_size*batch+self.batch_size]}
            y_true = self.train_y[self.batch_size*batch: self.batch_size*batch+self.batch_size]

            y_pred = self.model(inputs, training=False)
            # y_pred = K.argmax(y_pred, axis=1)

            loss = tf.reduce_mean(tf.math.squared_difference(y_pred, y_true))

            if batch % 10 == 0:
                print("loss = {}".format(loss))

            gradients = tape.gradient(loss, embeddings)  # Embedding梯度
            grads = tf.convert_to_tensor(gradients)
            self.delta = epsilon * grads / (K.sqrt(K.sum(grads ** 2)) + 1e-8)  # 计算扰动
            K.set_value(embeddings, K.eval(embeddings) + self.delta)  # 注入扰动

    def on_batch_end(self, batch, logs=None):
        # print("end batch: {}, logs: {}".format(batch, logs))
        # {'loss': 4.160139560699463, 'sparse_categorical_accuracy': 0.1899999976158142}
        if batch % 10 == 0:
            print("end batch: {}, logs: {}".format(batch, logs))
        embedding_name = 'Embedding-Token'
        embeddings = self.model.get_layer(embedding_name).embeddings
        K.set_value(embeddings, K.eval(embeddings) - self.delta)  # 删除扰动
        self.last_logs = logs

########################################################################################################################
from keras.engine import data_adapter

class AdversarialModel(keras.Model):
    def __init__(self, *args, **kwargs):
        super(AdversarialModel, self).__init__(*args, **kwargs)

    def train_step(self, data, epsilon=0.5):
        # 这里train_step修改，基于Model修改，不同keras版本可能有细微差别；

        # data = data_adapter.expand_1d(data)
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compute_loss(x, y, y_pred, sample_weight)

        embeddings = [var for var in self.trainable_variables if 'embedding' in var.name][0]

        # grads_and_vars = self.optimizer.compute_gradients(loss, embeddings, tape)  # List of (gradient, variable) pairs.
        grads = tape.gradient(loss, [embeddings])[0] # Embedding梯度
        grads = tf.convert_to_tensor(grads)
        # embeddings = tf.convert_to_tensor(embeddings)
        delta = epsilon * grads / (K.sqrt(K.sum(grads ** 2)) + 1e-8)  # 计算扰动
        embeddings = tf.convert_to_tensor(embeddings)
        embeddings = embeddings + delta # 注入扰动

        # Run forward pass.
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compute_loss(x, y, y_pred, sample_weight)
        self._validate_target_and_loss(y, loss)
        # Run backwards pass.
        self.optimizer.minimize(loss, self.trainable_variables,
                                tape=tape)  # 实现自动梯度下降minimize()操作也只是一个compute_gradients()和apply_gradients()的组合操作.
        # compute_gradients()用来计算梯度，opt.apply_gradients()用来更新参数。
        embeddings = embeddings - delta  # 删除扰动
        self.compiled_metrics.update_state(y, y_pred, sample_weight)
        return self.compute_metrics(x, y, y_pred, sample_weight)


model = AdversarialModel([*model.inputs], model.output)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics='accuracy')
model.summary()

train_x = {"word_ids": np.random.randint(1928, size=(100, 10, 79)),
           "weights": np.random.random(size=(100, 10)),
           }
train_y = np.random.randint(2, size=(100, 1))
model.fit(train_x, train_y, batch_size=32)
########################################################################################################################

def main():

    historys = model.fit(data_generator2(train_data, batch_size),
                                   steps_per_epoch=len(train_data)//batch_size,
                                   # 在声明一个 epoch 完成并开始下一个 epoch 之前从 generator 产生的总步数（批次样本）。 它通常应该等于你的数据集的样本数量除以批量大小。len(train_data)/batch_size
                                   validation_data=data_generator2(valid_data, batch_size),
                                   validation_steps=len(valid_data)//batch_size,
                                   # 仅当 validation_data 是一个生成器时才可用。 在停止前 generator 生成的总步数（样本批数）。len(dev_data)/batch_size
                                   epochs=50,
                                   verbose=1,
                                   shuffle=True,
                                   )

    train_x, train_y = data_generator3(train_data, batch_size)
    historys = model.fit(x=train_x, y=train_y,
                         validation_data=data_generator3(valid_data, batch_size),
                         batch_size=batch_size,
                         # 仅当 validation_data 是一个生成器时才可用。 在停止前 generator 生成的总步数（样本批数）。len(dev_data)/batch_size
                         epochs=1,
                         verbose=1,
                         shuffle=True,
                         callbacks=[GradientCalcCallback(train_x, train_y, model, batch_size)]
                         )
    print(historys.history)

    # model.load_weights('best_model.weights')
    predict_to_file(rf'D:/Users/{USERNAME}/data/iflytek/test.json', 'iflytek_predict.json')

if __name__ == '__main__':
    main()

