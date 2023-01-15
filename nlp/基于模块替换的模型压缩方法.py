#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 文本分类例子下的模型压缩
# 方法为BERT-of-Theseus
# 论文：https://arxiv.org/abs/2002.02925
# 来源：https://kexue.fm/archives/7575， https://github.com/bojone/bert4keras/examples

'''
模型压缩要花费更长时间的原因是它需要“先训练大模型，再压缩为小模型”。
为什么不直接训练一个小模型？答案是目前很多实验已经表明，先训练大模型再压缩，相比直接训练一个小模型，最后的精度通常会更高一些。

常见的模型压缩技术可以分为两大类：1、直接简化大模型得到小模型；2、借助大模型重新训练小模型。这两种手段的共同点是都先要训练出一个效果比较好的大模型，然后再做后续操作。

第一类的代表方法是剪枝（Pruning）和量化（Quantization）。剪枝，顾名思义，就是试图删减掉原来大模型的一些组件，使其变为一个小模型，同时使得模型效果在可接受的范围内；
至于量化，指的是不改变原模型结构，但将模型换一种数值格式，同时也不严重降低效果，通常我们建立和训练模型用的是float32类型，而换成float16就能提速且省显存，如果能进一步转换成8位整数甚至2位整数（二值化），那么提速省显存的效果将会更加明显。

第二类的代表方法是蒸馏（Distillation）。蒸馏的基本想法是将大模型的输出当作小模型训练时的标签来用，以分类问题为例，实际的标签是one hot形式的，
大模型的输出（比如logits）则包含更丰富的信号，所以小模型能从中学习到更好的特征。除了学习大模型的输出之外，很多时候为了更进一步提升效果，
还需要小模型学习大模型的中间层结果、Attention矩阵、相关矩阵等，所以一个好的蒸馏过程通常涉及到多项loss，如何合理地设计这些loss以及调整这些loss的权重，是蒸馏领域的研究主题之一。

本文将要介绍的压缩方法称为“BERT-of-Theseus”，属于上面说的两大类压缩方法的第二类，也就是说它也是借助大模型来训练小模型，只不过它是基于模块的可替换性来设计的。
假设我们有一个6层的BERT，我们直接用它在下游任务上微调，得到一个效果还不错的模型，我们称之为Predecessor（前辈）；
我们的目的是得到一个3层的BERT，它在下游任务中效果接近Predecessor，至少比直接拿BERT的前3层去微调要好（否则就白费力气了），这个小模型我们称为Successor（传承者）。

在BERT-of-Theseus的整个流程中，Predecessor的权重都被固定住。
6层的Predecessor被分为3个模块，跟Successor的3层模型一一对应，训练的时候，随机用Successor层替换掉Predecessor的对应模块，然后直接用下游任务的优化目标进行微调（只训练Successor的层）。
训练充分后，再把整个Successor单独分离出来，继续在下游任务中微调一会，直到验证集指标不再上升。
在实现的时候，事实上是类似Dropout的过程，同时执行Predecessor和Successor模型，并将两者对应模块的输出之一置零，然后求和、送入下一层中，即
由于ε非0即1（不作调整，各自0.5概率随机选效果就挺好了），所以每个分支其实就相当于只有一个模块被选择到。
由于每次的置零都是随机的，因此训练足够多的步数后，Successor的每个层都能被训练好。

BERT-of-Theseus的命名源于思想实验“忒修斯(Theseus)之船”：如果忒修斯的船上的木头被逐渐替换，直到所有的木头都不是原来的木头，那这艘船还是原来的那艘船吗？
'''

import os
os.environ['TF_KERAS'] = '1'
USERNAME = os.getenv("USERNAME")

import json
import numpy as np
from bert4keras.backend import keras, K
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam, extend_with_piecewise_linear_lr
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open
from keras.layers import Input, Lambda, Dense, Layer
from keras.models import Model

num_classes = 119
maxlen = 128
batch_size = 32

# BERT base
config_path = rf'D:/Users/{USERNAME}/data/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = rf'D:/Users/{USERNAME}/data/chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = rf'D:/Users/{USERNAME}/data/chinese_L-12_H-768_A-12/vocab.txt'


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


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text, label) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([label])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


# 转换数据集
train_generator = data_generator(train_data, batch_size)
valid_generator = data_generator(valid_data, batch_size)


class BinaryRandomChoice(Layer):
    """随机二选一
    """
    def __init__(self, **kwargs):
        super(BinaryRandomChoice, self).__init__(**kwargs)
        self.supports_masking = True

    def compute_mask(self, inputs, mask=None):
        if mask is not None:
            return mask[1]

    def call(self, inputs):
        source, target = inputs
        mask = K.random_binomial(shape=[1], p=0.5)
        output = mask * source + (1 - mask) * target
        return K.in_train_phase(output, target)

    def compute_output_shape(self, input_shape):
        return input_shape[1]


def bert_of_theseus(predecessor, successor, classfier):
    """bert of theseus
    """
    inputs = predecessor.inputs
    # 固定住已经训练好的层
    for layer in predecessor.model.layers:
        layer.trainable = False
    classfier.trainable = False
    # Embedding层替换
    predecessor_outputs = predecessor.apply_embeddings(inputs)
    successor_outputs = successor.apply_embeddings(inputs)
    outputs = BinaryRandomChoice()([predecessor_outputs, successor_outputs])
    # Transformer层替换
    layers_per_module = predecessor.num_hidden_layers // successor.num_hidden_layers
    for index in range(successor.num_hidden_layers):
        predecessor_outputs = outputs
        for sub_index in range(layers_per_module):
            predecessor_outputs = predecessor.apply_main_layers(
                predecessor_outputs, layers_per_module * index + sub_index
            )
        successor_outputs = successor.apply_main_layers(outputs, index)
        outputs = BinaryRandomChoice()([predecessor_outputs, successor_outputs])
    # 返回模型
    outputs = classfier(outputs)
    model = Model(inputs, outputs)
    return model


def evaluate(data, model):
    total, right = 0., 0.
    for x_true, y_true in data:
        y_pred = model.predict(x_true).argmax(axis=1)
        y_true = y_true[:, 0]
        total += len(y_true)
        right += (y_true == y_pred).sum()
    return right / total


class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """
    def __init__(self, savename):
        self.best_val_acc = 0.
        self.savename = savename

    def on_epoch_end(self, epoch, logs=None):
        val_acc = evaluate(valid_generator, self.model)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.model.save_weights(self.savename)
        print(
            u'val_acc: %.5f, best_val_acc: %.5f\n' %
            (val_acc, self.best_val_acc)
        )


# 加载预训练模型（12层）
predecessor = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    return_keras_model=False,
    prefix='Predecessor-'
)

# 加载预训练模型（3层）
successor = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    return_keras_model=False,
    num_hidden_layers=3,
    prefix='Successor-'
)

# 判别模型
x_in = Input(shape=K.int_shape(predecessor.output)[1:])
x = Lambda(lambda x: x[:, 0])(x_in)
x = Dense(units=num_classes, activation='softmax')(x)
classfier = Model(x_in, x)

predecessor_model = Model(predecessor.inputs, classfier(predecessor.output))
predecessor_model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=Adam(2e-5),  # 用足够小的学习率
    metrics=['sparse_categorical_accuracy'],
)
predecessor_model.summary()

successor_model = Model(successor.inputs, classfier(successor.output))
successor_model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=Adam(2e-5),  # 用足够小的学习率
    metrics=['sparse_categorical_accuracy'],
)
successor_model.summary()

theseus_model = bert_of_theseus(predecessor, successor, classfier)
theseus_model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=Adam(2e-5),  # 用足够小的学习率
    metrics=['sparse_categorical_accuracy'],
)
theseus_model.summary()


def main():

    # 训练predecessor
    predecessor_evaluator = Evaluator('best_predecessor.weights')
    predecessor_model.fit(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=5,
        callbacks=[predecessor_evaluator]
    )

    # 训练theseus
    theseus_evaluator = Evaluator('best_theseus.weights')
    theseus_model.fit(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=10,
        callbacks=[theseus_evaluator]
    )
    theseus_model.load_weights('best_theseus.weights')

    # 训练successor
    successor_evaluator = Evaluator('best_successor.weights')
    successor_model.fit(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=5,
        callbacks=[successor_evaluator]
    )


if __name__ == '__main__':
    main()
