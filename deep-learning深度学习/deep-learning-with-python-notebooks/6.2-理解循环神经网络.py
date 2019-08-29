#!/usr/bin/env python
# coding: utf-8

# 来源： https://github.com/fchollet/deep-learning-with-python-notebooks.gi
# In[1]:


import keras
keras.__version__


# 简单 RNN 的 Numpy 实现

import numpy as np
timesteps = 100 # 输入序列的时间步数
input_features = 32 # 输入特征空间的维度
output_features = 64 # 输出特征空间的维度



# 输入数据:随机噪声,
inputs = np.random.random((timesteps, input_features))
state_t = np.zeros((output_features,)) # 初始状态:全零向量

# 创建随机的权重矩阵
W = np.random.random((output_features, input_features))
U = np.random.random((output_features, output_features))
b = np.random.random((output_features,))


successive_outputs = []
# input_t 是形状为 (input_features,) 的向量
for input_t in inputs:
    # 由输入和当前状态(前一个输出)计算得到当前输出
    output_t = np.tanh(np.dot(W, input_t) + np.dot(U, state_t) + b)
    successive_outputs.append(output_t) # 将这个输出保存到一个列表中
    state_t = output_t # 更新网络的状态,用于下一个时间步

# 最终输出是一个形状为 (timesteps,output_features) 的二维张量
final_output_sequence = np.stack(successive_outputs, axis=0)



from keras.layers import SimpleRNN

# 上面 Numpy 的简单实现,对应一个实际的 Keras 层,即 SimpleRNN 层。
# from keras.layers import SimpleRNN
# 二者有一点小小的区别: SimpleRNN 层能够像其他 Keras 层一样处理序列批量,而不是
# 像 Numpy 示例那样只能处理单个序列。因此,它接收形状为 (batch_size, timesteps,
# input_features) 的输入,而不是 (timesteps, input_features) 。
# 与 Keras 中的所有循环层一样, SimpleRNN 可以在两种不同的模式下运行:一种是返回每
# 个时间步连续输出的完整序列,即形状为 (batch_size, timesteps, output_features)
# 的三维张量;另一种是只返回每个输入序列的最终输出,即形状为 (batch_size, output_
# features) 的二维张量。这两种模式由 return_sequences 这个构造函数参数来控制。我们
# 来看一个使用 SimpleRNN 的例子,它只返回最后一个时间步的输出。


from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN

model = Sequential()
model.add(Embedding(10000, 32))
model.add(SimpleRNN(32))
model.summary()


# In[4]:


model = Sequential()
model.add(Embedding(10000, 32))
model.add(SimpleRNN(32, return_sequences=True))
model.summary()

'''
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding_1 (Embedding)      (None, None, 32)          320000    = 10000 * 32
_________________________________________________________________
simple_rnn_1 (SimpleRNN)     (None, 32)                2080      = 32 * (32 + 32) + 32
=================================================================
Total params: 322,080
Trainable params: 322,080
Non-trainable params: 0
_________________________________________________________________
g, 一个单元中的FFNNs的数量（RNN有1个，GRU有3个，LSTM有4个）
h, 隐藏单元的大小
i,输入的维度/大小 
因为每一个FFNN有h(h+i)+h个参数，则我们有
参数数量=g×[h(h+i)+h]
'''
# 为了提高网络的表示能力,将多个循环层逐个堆叠有时也是很有用的。在这种情况下,你
# 需要让所有中间层都返回完整的输出序列。
# In[5]:


model = Sequential()
model.add(Embedding(10000, 32))
model.add(SimpleRNN(32, return_sequences=True))
model.add(SimpleRNN(32, return_sequences=True))
model.add(SimpleRNN(32, return_sequences=True))
model.add(SimpleRNN(32))  # 最后一层仅返回最终输出
model.summary()

'''
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding_1 (Embedding)      (None, None, 32)          320000    10000 * 32
_________________________________________________________________
simple_rnn_1 (SimpleRNN)     (None, None, 32)          2080      = 32 * (32+32) + 32
_________________________________________________________________
simple_rnn_2 (SimpleRNN)     (None, None, 32)          2080      = 32 * (32+32) + 32
_________________________________________________________________
simple_rnn_3 (SimpleRNN)     (None, None, 32)          2080      = 32 * (32+32) + 32
_________________________________________________________________
simple_rnn_4 (SimpleRNN)     (None, 32)                2080      = 32 * (32+32) + 32
=================================================================
Total params: 328,320
Trainable params: 328,320
Non-trainable params: 0
_________________________________________________________________

'''
# 接下来,我们将这个模型应用于 IMDB 电影评论分类问题。首先,对数据进行预处理。

# In[6]:


from keras.datasets import imdb
from keras.preprocessing import sequence

max_features = 10000  # 作为特征的单词个数
maxlen = 500  # 在这么多单词之后截断文本(这些单词都属于前 max_features 个最常见的单词)
batch_size = 32

print('Loading data...')
(input_train, y_train), (input_test, y_test) = imdb.load_data(num_words=max_features)
print(len(input_train), 'train sequences')
print(len(input_test), 'test sequences')

print('Pad sequences (samples x time)')
input_train = sequence.pad_sequences(input_train, maxlen=maxlen)
input_test = sequence.pad_sequences(input_test, maxlen=maxlen)
print('input_train shape:', input_train.shape)
print('input_test shape:', input_test.shape)


# 我们用一个 Embedding 层和一个 SimpleRNN 层来训练一个简单的循环网络。

# In[7]:


from keras.layers import Dense

model = Sequential()
model.add(Embedding(max_features, 32))
model.add(SimpleRNN(32))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(input_train, y_train,
                    epochs=10,
                    batch_size=128,
                    validation_split=0.2)


# 接下来显示训练和验证的损失和精度

# In[8]:


import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()


# 提醒一下,在第 3 章,处理这个数据集的第一个简单方法得到的测试精度是 88%。不幸的是,
# 与这个基准相比,这个小型循环网络的表现并不好(验证精度只有 85%)。问题的部分原因在于,
# 输入只考虑了前 500 个单词,而不是整个序列,因此,RNN 获得的信息比前面的基准模型更少。
# 另一部分原因在于, SimpleRNN 不擅长处理长序列,比如文本。
# 其他类型的循环层的表现要好得多。我们来看几个更高级的循环层。

# [...]
# 
#Keras 中一个 LSTM 的具体例子
# 现在我们来看一个更实际的问题:使用 LSTM 层来创建一个模型,然后在 IMDB 数据上
# 训练模型。这个网络与前面介绍的 SimpleRNN 网络类似。你只需指定
# LSTM 层的输出维度,其他所有参数(有很多)都使用 Keras 默认值。Keras 具有很好的默认值,
# 无须手动调参,模型通常也能正常运行。
# In[11]:


from keras.layers import LSTM

model = Sequential()
model.add(Embedding(max_features, 32))
model.add(LSTM(32))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])
history = model.fit(input_train, y_train,
                    epochs=10,
                    batch_size=128,
                    validation_split=0.2)


# In[12]:


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

