#!/usr/bin/env python
# coding=utf-8

import os 
import numpy as np 
import pandas as pd 
USERNAME = os.getenv("USERNAME")
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras import regularizers
import tensorflow as tf
from tensorflow.keras.layers import Input,Embedding,LSTM,Dense,Dropout,BatchNormalization,Reshape, Flatten, Concatenate, Lambda, Add, Conv2D, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from tensorflow import keras 
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.utils import shuffle

class ArcFace(Layer):
    def __init__(self, n_classes=10, s=30.0, m=0.50, regularizer=None, **kwargs):
        super(ArcFace, self).__init__(**kwargs)
        self.n_classes = n_classes
        self.s = s
        self.m = m
        self.regularizer = regularizers.get(regularizer)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'n_classes': self.n_classes,
            's': self.s,
            'm': self.m,
            'regularizer': self.regularizer
        })
        return config

    def build(self, input_shape):
        super(ArcFace, self).build(input_shape[0])
        self.W = self.add_weight(name='W',
                                 shape=(input_shape[0][-1], self.n_classes),
                                 initializer='glorot_uniform',
                                 trainable=True,
                                 regularizer=self.regularizer)

    def call(self, inputs):
        x, y = inputs
        c = K.shape(x)[-1]
        # normalize feature
        x = tf.nn.l2_normalize(x, axis=1)
        # normalize weights
        W = tf.nn.l2_normalize(self.W, axis=0)
        # dot product
        logits = x @ W
        # add margin
        # clip logits to prevent zero division when backward
        theta = tf.acos(K.clip(logits, -1.0 + K.epsilon(), 1.0 - K.epsilon()))
        target_logits = tf.cos(theta + self.m)
        # sin = tf.sqrt(1 - logits**2)
        # cos_m = tf.cos(logits)
        # sin_m = tf.sin(logits)
        # target_logits = logits * cos_m - sin * sin_m
        #
        logits = logits * (1 - y) + target_logits * y
        # feature re-scale
        logits *= self.s
        out = tf.nn.softmax(logits)
        return out

    def compute_output_shape(self, input_shape):
        return (None, self.n_classes)


class SphereFace(Layer):
    def __init__(self, n_classes=10, s=30.0, m=1.35, regularizer=None, **kwargs):
        super(SphereFace, self).__init__(**kwargs)
        self.n_classes = n_classes
        self.s = s
        self.m = m
        self.regularizer = regularizers.get(regularizer)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'n_classes': self.n_classes,
            's': self.s,
            'm': self.m,
            'regularizer': self.regularizer
        })
        return config

    def build(self, input_shape):
        super(SphereFace, self).build(input_shape[0])
        self.W = self.add_weight(name='W',
                                 shape=(input_shape[0][-1], self.n_classes),
                                 initializer='glorot_uniform',
                                 trainable=True,
                                 regularizer=self.regularizer)

    def call(self, inputs):
        x, y = inputs
        c = K.shape(x)[-1]
        # normalize feature
        x = tf.nn.l2_normalize(x, axis=1)
        # normalize weights
        W = tf.nn.l2_normalize(self.W, axis=0)
        # dot product
        logits = x @ W
        # add margin
        # clip logits to prevent zero division when backward
        theta = tf.acos(K.clip(logits, -1.0 + K.epsilon(), 1.0 - K.epsilon()))
        target_logits = tf.cos(self.m * theta)
        #
        logits = logits * (1 - y) + target_logits * y
        # feature re-scale
        logits *= self.s
        out = tf.nn.softmax(logits)
        return out

    def compute_output_shape(self, input_shape):
        return (None, self.n_classes)


class CosFace(Layer):
    def __init__(self, n_classes=10, s=30.0, m=0.35, regularizer=None, **kwargs):
        super(CosFace, self).__init__(**kwargs)
        self.n_classes = n_classes
        self.s = s
        self.m = m
        self.regularizer = regularizers.get(regularizer)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'n_classes': self.n_classes,
            's': self.s,
            'm': self.m,
            'regularizer': self.regularizer
        })
        return config

    def build(self, input_shape):
        super(CosFace, self).build(input_shape[0])
        self.W = self.add_weight(name='W',
                                 shape=(input_shape[0][-1], self.n_classes),
                                 initializer='glorot_uniform',
                                 trainable=True,
                                 regularizer=self.regularizer)

    def call(self, inputs):
        x, y = inputs
        c = K.shape(x)[-1]
        # normalize feature
        x = tf.nn.l2_normalize(x, axis=1)
        # normalize weights
        W = tf.nn.l2_normalize(self.W, axis=0)
        # dot product
        logits = x @ W
        # add margin
        target_logits = logits - self.m
        #
        logits = logits * (1 - y) + target_logits * y
        # feature re-scale
        logits *= self.s
        out = tf.nn.softmax(logits)
        return out

    def compute_output_shape(self, input_shape):
        return (None, self.n_classes)

def test1():
    input = Input(shape=(28, 28, 1))
    label = Input(shape=(10,))
    x = Conv2D(32, kernel_size=(3, 3), activation='relu')(input)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Flatten()(x)
    x = Dense(512, kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    output = ArcFace(n_classes=10)([x, label])
    model = Model([input, label], output)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(),
                  metrics=['accuracy'])
    
    print(model.summary())
    
    model.fit([x_train, y_train],
              y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=([x_test, y_test], y_test),
              callbacks=[ModelCheckpoint('model.hdf5',
                         verbose=1, save_best_only=True)])
    
    
    model.load_weights('model.hdf5')
    model = Model(inputs=model.input[0], outputs=model.layers[-3].output)
    embedded_features = model.predict(x_test, verbose=1)
    embedded_features /= np.linalg.norm(embedded_features, axis=1, keepdims=True)
    
    
    # 资料来源：https://github.com/4uiiurz1/keras-arcface

def test2():
    '''中文新闻分类'''
    data_file = rf"D:\Users\{USERNAME}/github_project/toutiao-text-classfication-dataset/toutiao_cat_data.txt"  # 数据来源： http://github.com/skdjfla/toutiao-text-classfication-dataset
    df = pd.read_csv(data_file, sep='_!_', names=['id', 'code', 'label', 'title', 'keyword'], engine='python')
    df = shuffle(df)
    texts = [' '.join(list(text)) for text in df['title'].values]
    # texts[:3]
    # Out[128]:
    # array(['京城最值得你来场文化之旅的博物馆', '发酵床的垫料种类有哪些？哪种更好？', '上联：黄山黄河黄皮肤黄土高原。怎么对下联？'],
    #       dtype=object)
    tokenizer = Tokenizer(num_words=10000)
    # 根据文本列表更新内部词汇表。
    tokenizer.fit_on_texts(texts)

    # 将文本中的每个文本转换为整数序列。
    # 只有最常出现的“num_words”字才会被考虑在内。
    # 只有分词器知道的单词才会被考虑在内。
    sequences = tokenizer.texts_to_sequences(texts)
    # dict {word: index}
    word_index = tokenizer.word_index

    print('tokens数量：', len(word_index))

    data = pad_sequences(sequences, maxlen=48)
    print('Shape of data tensor:', data.shape)

    code2id = {label: idx for idx, label in enumerate(sorted(df['label'].unique()))}
    y = [code2id[label] for label in df['label'].values]
    
    x_train, x_test, y_train, y_test = train_test_split(data, y, test_size=0.15)
    del data, y, texts, sequences
    
    input_shape = x_train.shape[1:]
    num_classes = len(code2id)
    y_train = keras.utils.to_categorical(y_train, num_classes=num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes=num_classes)
    
    # 开始构建模型
    input = Input(shape=input_shape)
    label = Input(shape=(num_classes,))
    
    x = Embedding(10000, 64)(input)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.2)(x)
    x = Dense(64, activation="relu")(x)

    x = Dropout(0.2)(x)
    x = Flatten()(x)
    x = Dense(128, kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    output = ArcFace(n_classes=num_classes)([x, label])
    model = Model([input, label], output)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(),
                  metrics=['accuracy'])

    print(model.summary())
    batch_size = 16
    epochs = 3
    model.fit([x_train, y_train],
              y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_split = 0.1,
              # validation_data=([x_test, y_test], y_test),
              callbacks=[ModelCheckpoint('./models/model.hdf5', verbose=1, save_best_only=True)]
              )

    # 加载模型获取文本特征；
    model.load_weights('./models/model.hdf5')
    
    model = Model(inputs=model.input[0], outputs=model.layers[-3].output)
    embedded_features = model.predict(x_test, verbose=1)
    embedded_features /= np.linalg.norm(embedded_features, axis=1, keepdims=True)


    # CenterLoss和ArcLoss都只是在训练时提高提取器的特征提取能力，在做识别模型时使用的是他们之前的特征feature，在做分类时才会使用通过他们激活之后的分类。
    
    x_test_feat = embedded_features
    embedded_features = model.predict(x_train, verbose=1)
    embedded_features /= np.linalg.norm(embedded_features, axis=1, keepdims=True)
    x_train_feat = embedded_features 
    
    model = Sequential()
    model.add(Dense(64, activation='relu', input_dim=x_train_feat.shape[-1]))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])

    model.fit(x_train_feat, y_train,
              epochs=20,
              batch_size=8)
    score = model.evaluate(x_test_feat, y_test, batch_size=8)
    print(score)

    # 先通过arcFace获取特征向量，再基于特征向量进行文本分类，貌似效果不好；原因有待分析；
    
def main():
    pass


if __name__ == "__main__":
    main()
