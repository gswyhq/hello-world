#!/usr/bin/env python
# coding=utf-8

import os 
from sentence_transformers import SentenceTransformer
USERNAME = os.getenv("USERNAME")


import os 
import numpy as np 
import pandas as pd 
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras import regularizers
import tensorflow as tf
from tensorflow.keras.layers import Input,Embedding,LSTM,Dense,Dropout,BatchNormalization,Reshape, Flatten, Concatenate, Lambda, Add, Conv2D, MaxPooling2D, LSTM
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from tensorflow import keras 
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.utils import shuffle

sentences_1 = ["样例数据-1", "样例数据-2"]
sentences_2 = ["样例数据-3", "样例数据-4"]
# 模型来源：https://www.modelscope.cn/models/AI-ModelScope/bge-small-zh-v1.5/files
model_dir = rf"D:\Users\{USERNAME}\data\bge-small-zh-v1.5"
model = SentenceTransformer(model_dir)
embeddings_1 = model.encode(sentences_1, normalize_embeddings=True)
embeddings_2 = model.encode(sentences_2, normalize_embeddings=True)
similarity = embeddings_1 @ embeddings_2.T
print(similarity)

data_file = rf"D:\Users\{USERNAME}/github_project/toutiao-text-classfication-dataset/toutiao_cat_data.txt"  # 数据来源： http://github.com/skdjfla/toutiao-text-classfication-dataset
df = pd.read_csv(data_file, sep='_!_', names=['id', 'code', 'label', 'title', 'keyword'], engine='python')
df = shuffle(df)

data = model.encode(df['title'].values, normalize_embeddings=True)
code2id = {label: idx for idx, label in enumerate(sorted(df['label'].unique()))}
y = [code2id[label] for label in df['label'].values]
num_classes = len(code2id)
x_train, x_test, y_train, y_test = train_test_split(data, y, test_size=0.15)

y_train = keras.utils.to_categorical(y_train, num_classes=num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes=num_classes)

model = Sequential()
model.add(Dense(64, activation='relu', input_dim=x_train.shape[-1]))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

model.fit(x_train, y_train,
          epochs=2,
          batch_size=8)
loss, accuracy = model.evaluate(x_test, y_test, batch_size=8)
print(loss, accuracy)

# Epoch 1/2
# 40661/40661 [==============================] - 26s 622us/step - loss: 0.7516 - accuracy: 0.7787
# Epoch 2/2
# 40661/40661 [==============================] - 25s 618us/step - loss: 0.6429 - accuracy: 0.8112
# 7176/7176 [==============================] - 4s 514us/step - loss: 0.5405 - accuracy: 0.8368
# [0.5405106544494629, 0.8367883563041687]

# 结论：直接使用预训练模型的输出作为特征向量进行文本分类；效果test_acc: 83.68%

###########################################################################################################################
# 从0训练效果对比：
texts = [' '.join(list(text)) for text in df['title'].values]
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
num_classes = len(code2id)

x_train, x_test, y_train, y_test = train_test_split(data, y, test_size=0.15)
del data, y, texts, sequences

input_shape = x_train.shape[1:]

y_train = keras.utils.to_categorical(y_train, num_classes=num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes=num_classes)

vocab_size = 10000
embedding_dim = 32
max_length = 48
model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, input_length=max_length))
model.add(LSTM(128))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, batch_size=16, validation_split=0.1)
loss, accuracy = model.evaluate(x_test, y_test, batch_size=8)
print(loss, accuracy)
# 0.5342428684234619 0.8452372550964355

# 对比预训练模式的acc结果，两者差不多；由此可见，使用预训练模型并没有什么作用；也许是该预训练模型不适用此任务；也许是使用方法不对；

###########################################################################################################################
# 对比bert
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
# Out[29]: '2.9.0'
# tf.__version__
# Out[30]: '2.9.1'
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

data_file = rf"D:\Users\{USERNAME}/github_project/toutiao-text-classfication-dataset/toutiao_cat_data.txt"  # 数据来源： http://github.com/skdjfla/toutiao-text-classfication-dataset
df = pd.read_csv(data_file, sep='_!_', names=['id', 'code', 'label', 'title', 'keyword'], engine='python', usecols=['title', 'label'])
df = shuffle(df)

data = df['title'].values
code2id = {label: idx for idx, label in enumerate(sorted(df['label'].unique()))}
y = [code2id[label] for label in df['label'].values]
num_classes = len(code2id)
x_train, x_test, y_train, y_test = train_test_split(data, y, test_size=0.15)

y_train = keras.utils.to_categorical(y_train, num_classes=num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes=num_classes)

x_train_indices_segments = np.array([tokenizer.encode(text, second=None, max_len=48) for text in x_train])
x_test_indices_segments = np.array([tokenizer.encode(text, second=None, max_len=48) for text in x_test])

x_train_indices, x_train_segments = x_train_indices_segments[:,0,:], x_train_indices_segments[:,1,:]
x_test_indices, x_test_segments = x_test_indices_segments[:,0,:], x_test_indices_segments[:,1,:]

bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=None)
for l in bert_model.layers:
    l.trainable = True
    
x1_in = Input(shape=(None,))
x2_in = Input(shape=(None,))
x = bert_model([x1_in, x2_in])
x = Lambda(lambda x: x[:, 0])(x)
x = Dense(64, activation='relu')(x)
p = Dense(15, activation='softmax')(x)
model = Model([x1_in, x2_in], p)
model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=5e-5), # 用足够小的学习率
    metrics=['accuracy']
)
model.summary()

model.fit([x_train_indices, x_train_segments], y_train, epochs=3, batch_size=16, validation_split=0.1)
loss, accuracy = model.evaluate([x_test_indices, x_test_segments], y_test, batch_size=8)
print(loss, accuracy)

# 18298/18298 [==============================] - 6255s 341ms/step - loss: 0.6379 - accuracy: 0.8124 - val_loss: 0.5027 - val_accuracy: 0.8503
# Epoch 2/3
# 18298/18298 [==============================] - 6620s 362ms/step - loss: 0.4703 - accuracy: 0.8605 - val_loss: 0.4725 - val_accuracy: 0.8625
# Epoch 2/3
# 18298/18298 [==============================] - 6108s 334ms/step - loss: 0.4043 - accuracy: 0.8794 - val_loss: 0.4433 - val_accuracy: 0.8717
# Epoch 3/3
# 18298/18298 [==============================] - 6125s 335ms/step - loss: 0.3549 - accuracy: 0.8929 - val_loss: 0.4417 - val_accuracy: 0.8741
# 7176/7176 [==============================] - 421s 59ms/step - loss: 0.4508 - accuracy: 0.8714
# 0.4507611095905304 0.8714375495910645

# 结论：使用预训练模型，并需要微调预训练模型参数，效果才有所提升；最终测试集正确率达到87%

def main():
    pass


if __name__ == "__main__":
    main()

