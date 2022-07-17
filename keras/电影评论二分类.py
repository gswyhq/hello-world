#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pylab
from pandas import DataFrame, Series

#本节使用 IMDB 数据集，它包含来自互联网电影数据库（IMDB）的 50 000 条严重两极分化的评论。数据集被分为用于训练的 25 000 条评论与用于测试的 25 000 条评论，训练集和测试 集都包含 50% 的正面评论和 50% 的负面评论。

#它已经过预处理：评论（单词序列） 已经被转换为整数序列，其中每个整数代表字典中的某个单词

from keras.datasets import imdb

# 下载imdb 数据，也可以事先下载好
# Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/imdb.npz
# https://storage.googleapis.com/tensorflow/tf-keras-datasets/imdb_word_index.json
# md5sum ~/.keras/datasets/imdb.npz
# 599dadb1135973df5b59232a0e9a887c  /home/mobaxterm/.keras/datasets/imdb.npz

(train_data, train_labels), (test_data, test_labels) = imdb.load_data( num_words=10000)
#参数 num_words=10000 的意思是仅保留训练数据中前 10000 个最常出现的单词。低频单 词将被舍弃。这样得到的向量数据不会太大，便于处理。


print(train_data[0],train_labels[0])

print(max([max(sequence) for sequence in train_data]))#9999

word_index=imdb.get_word_index()
reverse_word_index=dict([(v,k) for k,v in word_index.items()])
decoded_review=' '.join([reverse_word_index.get(i-3,'?') for i in train_data[0]])#i-3 因为 0、1、2 是为“padding”（填充）、“start of sequence”（序列开始）、“unknown”（未知词）分别保留的索引
print(decoded_review)

#准备数据：
# 对列表进行 one-hot 编码，将其转换为 0 和 1 组成的向量。举个例子，序列[3, 5]将会 被转换为 10000 维向量，只有索引为 3 和 5 的元素是 1，其余元素都是 0。然后网络第一层可以用 Dense 层，它能够处理浮点数向量数据
#将整数序列编码为二进制矩阵
def sequences2vector(sequences,dimension=10000):
    results=np.zeros((len(sequences),dimension))#创建一个shape(len(sequeces),10000)的零矩阵
    print(results.shape)
    for i,sequence in enumerate(sequences):#枚举（index,sequence）
        results[i,sequence]=1

    return results
x_train=sequences2vector(train_data)
x_test=sequences2vector(test_data)

#标签向量化
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

#构建网络
'''
输入数据是向量，而标签是标量（1 和 0），这是你会遇到的最简单的情况。有一类网络在这种问题上表现很好，就是带有 relu 激活的全连接层（Dense）的简单堆叠，比如
    Dense(16, activation='relu')#传入Dense层的参数（16）是该层隐藏单元的个数。一个隐藏单元（hidden unit）是该层表示空间的一个维度

每个带有 relu 激活的 Dense 层都实现了下列张量运算：
     output = relu(dot(W, input) + b)

隐藏单元越多（即更高维的表示空间），网络越能够学到更加复杂的表示，但网络的计算代价也变得更大，而且可能会导 致学到不好的模式（这种模式会提高训练数据上的性能，但不会提高测试数据上的性能）

    对于这种 Dense 层的堆叠，你需要确定以下两个关键架构：
        网络有多少层； 
        每层有多少个隐藏单元
'''
#这里选择，两个中间层，每层有16个隐藏单元；第三层输出一个标量，预测当前评论的情感【中间层使用relu作为激活函数，最后一层使用sigmoid激活函数以输出一个0-1范围内的概率值(表示样本目标值等于1的可能性)】
#relu（rectified linear unit， 整流线性单元）函数将所有负值归零
# sigmoid 函数则将任意值“压缩”到 [0, 1] 区间内，其输出值可以看作概率值

#模型定义
from keras import models
from keras import layers
from keras import regularizers

model=models.Sequential()
model.add(layers.Dense(16,activation='relu',input_shape=(10000,)))#未指定第0轴批量
model.add(layers.Dense(16,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))
#为了得到更丰富的假设空间，从而充分利用多层表示的优势，你需要添加非线性或激活函数。relu 是深度学习中最常用的激活函数，但还有许多其他函数可选，它们都有类似 的奇怪名称，比如 prelu、elu 等

#选择损失函数和优化器
#你面对的是一个二分类问题，网络输出是一个概率值（网络最后一层使用 sigmoid 激活函数，仅包含一个单元），那么最好使用 binary_ crossentropy（二元交叉熵）损失。这并不是唯一可行的选择，比如你还可以使用 mean_ squared_error（均方误差）。但对于输出概率值的模型，交叉熵（crossentropy）往往是最好 的选择。交叉熵是来自于信息论领域的概念，用于衡量概率分布之间的距离，在这个例子中就 是真实分布与预测值之间的距离
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy']
              )
#配置自定义优化器的参数
from keras import optimizers

model.compile(optimizer=optimizers.RMSprop(lr=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

#传入自定义的损失函数或指标函数
from keras import losses,metrics
model.compile(optimizer=optimizers.RMSprop(lr=0.001),
              loss=losses.binary_crossentropy,
              metrics=[metrics.binary_accuracy])

#留出验证集【为了在训练过程中监控模型在前所未见的数据上的精度，将原始训练数据留出 10 000 个样本作为验证集】
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

#训练模型
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc']
              )
'''
使用 512 个样本组成的小批量，将模型训练 20 个轮次（即对 x_train 和 y_train 两个张量中的所有样本进行20次迭代）。与此同时，为了监控在留出的 10000个样本上的损失和精度。你可以通过将验证数据传入validation_data 参数来完成
'''
history=model.fit(partial_x_train,
                  partial_y_train,
                  batch_size=512,
                  epochs=20,
                  validation_data=(x_val,y_val))

history_dict=history.history#返回一个字典，训练过程中的所有数据
print(history_dict.keys())

#------------------------------------------------
'''
一种常见的降低过拟合的方法就是强制让模型权重只能取较小的值， 从而限制模型的复杂度，这使得权重值的分布更加规则（regular）。这种方法叫作权重正则化（weight regularization），其实现方法是向网络损失函数中添加与较大权重值相关的成本（cost）。 这个成本有两种形式。 
? L1 正则化（L1 regularization）：添加的成本与权重系数的绝对值［权重的 L1 范数（norm）］ 成正比。
? L2 正则化（L2 regularization）：添加的成本与权重系数的平方（权重的 L2 范数）成正比。 神经网络的 L2 正则化也叫权重衰减（weight decay）。不要被不同的名称搞混，权重衰减 与 L2 正则化在数学上是完全相同的
'''
#添加dropout正则化
#神经网络最有效也最常用的正则化方法之一
#对某一层使用 dropout，就是在训练过程中随机将该层的一些输出特征舍 弃（设置为 0）
#dropout 比率（dropout rate）是被设为 0 的特征所占的比例，通常在 0.2~0.5 范围内。测试时没有单元被舍弃，而该层的输出值需要按 dropout 比率缩小，因为这时比训练时 有更多的单元被激活，需要加以平衡

# 防止神经网络过拟合的常用方法包括：
#
# ?获取更多的训练数据
# ? 减小网络容量
# ? 添加权重正则化
# ? 添加 dropout

model3=models.Sequential()
model3.add(layers.Dense(16,kernel_regularizer=regularizers.l2(0.001),activation='relu',input_shape=(10000,)))#未指定第0轴批量
model3.add(layers.Dense(16,kernel_regularizer=regularizers.l2(0.001),activation='relu'))
model3.add(layers.Dense(1,activation='sigmoid'))

model3.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc']
              )

history3=model3.fit(partial_x_train,
                  partial_y_train,
                  batch_size=512,
                  epochs=20,
                  validation_data=(x_val,y_val))

history_dict3=history3.history
#-----------------------------------------------------------------
model4=models.Sequential()
model4.add(layers.Dense(16,kernel_regularizer=regularizers.l2(0.001),activation='relu',input_shape=(10000,)))#未指定第0轴批量
model4.add(layers.Dropout(0.5))
model4.add(layers.Dense(16,kernel_regularizer=regularizers.l2(0.001),activation='relu'))
model4.add(layers.Dropout(0.5))
model4.add(layers.Dense(1,activation='sigmoid'))

model4.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc']
              )

history4=model4.fit(partial_x_train,
                  partial_y_train,
                  batch_size=512,
                  epochs=20,
                  validation_data=(x_val,y_val))

history_dict4=history4.history

#--------------------------------------------

fig=plt.figure()
ax1=fig.add_subplot(1,1,1)
x=[i for i in range(1,21)]
# loss=history_dict['loss']
val_loss=history_dict['val_loss']
val_loss3=history_dict3['val_loss']
val_loss4=history_dict4['val_loss']

# ax1.plot(x,loss,'bo',label='Training loss')
ax1.plot(x,val_loss,'bo',label='validation loss')
ax1.plot(x,val_loss3,'ro',label='validation loss reg')
ax1.plot(x,val_loss4,'go',label='validation loss reg drop')
ax1.set_title('Training and validation loss')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
ax1.legend()

# ax2=fig.add_subplot(2,1,2)
# acc=history_dict['acc']
# val_acc = history_dict['val_acc']
# ax2.plot(x, acc, 'bo', label='Training acc')
# ax2.plot(x, val_acc, 'b', label='Validation acc')
# ax2.set_title('Training and validation accuracy')
# ax2.set_xlabel('Epochs')
# ax2.set_ylabel('Accuracy')
# ax2.legend()
plt.tight_layout()
plt.show()


#如图所示，训练损失每轮都在降低，训练精度每轮都在提升。这就是梯度下降优化的预期结果——你想要最小化的量随着每次迭代越来越小。
# 但验证损失和验证精度并非如此：它们似乎在第三/四轮达到最佳值。这就是我们之前警告过的一种情况：模型在训练数据上的表现越来越好，但在前所未见的数据上不一定表现得越来越好。

# 准确地说，你看到的是过拟合（overfit）：在第二轮之后，对训练数据过度优化，最终学到的表示仅针对于训练数据，无法泛化到训练集之外的数据

####这时可以在3轮之后停止训练，另外就是用一些方法降低过拟合

#从头开始训练一个模型，将epochs设置为4
model2=models.Sequential()
model2.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model2.add(layers.Dense(16, activation='relu'))
model2.add(layers.Dense(1, activation='sigmoid'))
model2.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model2.fit(x_train, y_train, epochs=4, batch_size=512)
results = model2.evaluate(x_test, y_test)
print(results)

#训练好网络之后，将其应用于实践，用predict方法得到评论为正面的可能性大小
temp=model2.predict(x_test)
print(temp)#可见，网络对某些样本的结果非常确信（大于等于 0.99，或小于等于 0.01），但对其他结果却不那么确信（0.6 或 0.4）

#对比实验
'''
通过以下实验，你可以确信前面选择的网络架构是非常合理的，虽然仍有改进的空间。
前面使用了两个隐藏层。你可以尝试使用一个或三个隐藏层，然后观察对验证精度和测试精度的影响。
尝试使用更多或更少的隐藏单元，比如 32 个、64 个等。 
尝试使用 mse 损失函数代替 binary_crossentropy。 
尝试使用 tanh 激活（这种激活在神经网络早期非常流行）代替 relu。

'''

'''
? 通常需要对原始数据进行大量预处理，以便将其转换为张量输入到神经网络中。单词序 列可以编码为二进制向量，但也有其他编码方式。
? 带有 relu 激活的 Dense 层堆叠，可以解决很多种问题（包括情感分类），你可能会经 常用到这种模型。
? 对于二分类问题（两个输出类别），网络的最后一层应该是只有一个单元并使用 sigmoid 激活的 Dense 层，网络输出应该是 0~1 范围内的标量，表示概率值。
? 对于二分类问题的 sigmoid 标量输出，你应该使用 binary_crossentropy 损失函数。 
? 无论你的问题是什么，rmsprop 优化器通常都是足够好的选择。这一点你无须担心。 
? 随着神经网络在训练数据上的表现越来越好，模型最终会过拟合，并在前所未见的数据 上得到越来越差的结果。一定要一直监控模型在训练集之外的数据上的性能。

'''
#

def main():
    pass


if __name__ == '__main__':
    main()