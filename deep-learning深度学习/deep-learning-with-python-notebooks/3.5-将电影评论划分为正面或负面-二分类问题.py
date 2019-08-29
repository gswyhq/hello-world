#!/usr/bin/env python
# coding: utf-8

# 来源： https://github.com/fchollet/deep-learning-with-python-notebooks.gi
# In[20]:


import keras
keras.__version__  # '2.2.4'

from keras.datasets import imdb

# 下列代码将会加载 IMDB 数据集(第一次运行时会下载大约 80MB 的数据)。
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# 参数 num_words=10000 的意思是仅保留训练数据中前 10 000 个最常出现的单词。
# 低频单词将被舍弃。这样得到的向量数据不会太大,便于处理。
# train_data 和 test_data 这两个变量都是评论组成的列表,每条评论又是单词索引组成的列表(表示一系列单词)。
# train_labels 和 test_labels 都是 0 和 1 组成的列表,其中 0代表负面(negative),1 代表正面(positive)。

train_data[0]
# [1, 14, 22, 16, ... 178, 32]

train_labels[0]
# 1

# 由于限定为前 10 000 个最常见的单词,单词索引都不会超过 10 000。
max([max(sequence) for sequence in train_data])


# 将某条评论迅速解码为英文单词。
# word_index 是一个将单词映射为整数索引的字典
word_index = imdb.get_word_index()

# 键值颠倒,将整数索引映射为单词
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

# 将评论解码。注意,索引减去了 3,因为 0、1、2是为“padding”(填充)、“start of sequence”(序列开始)、“unknown”(未知词)分别保留的索引
decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])


# In[26]:


decoded_review

# In[27]:


import numpy as np

def vectorize_sequences(sequences, dimension=10000):
    # 创建一个形状为 (len(sequences), dimension) 的零矩阵
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.  # 将 results[i] 的指定索引设为 1
    return results

# 将训练数据向量化
x_train = vectorize_sequences(train_data)
# 将测试数据向量化
x_test = vectorize_sequences(test_data)


# 样本现在变成了这样:
# >>> x_train[0]
# array([ 0., 1.,1., ...,0.,0.,0.])

# 将标签向量化
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')


# 输入数据是向量,而标签是标量(1 和 0),这是你会遇到的最简单的情况。
# 有一类网络在这种问题上表现很好,就是带有relu激活的全连接层(Dense)的简单堆叠,比如Dense(16, activation='relu') 。
# 传入 Dense 层的参数(16)是该层隐藏单元的个数。
# 一个隐藏单元(hidden unit)是该层表示空间的一个维度。
# 每个带有 relu 激活的 Dense 层都实现了下列张量运算:
# output = relu(dot(W, input) + b)
# 16 个隐藏单元对应的权重矩阵 W 的形状为 (input_dimension, 16) ,与 W 做点积相当于
# 将输入数据投影到 16 维表示空间中(然后再加上偏置向量 b 并应用 relu 运算)。
# 隐藏单元越多(即更高维的表示空间),网络越能够学到更加复杂的表示,
# 但网络的计算代价也变得更大,而且可能会导致学到不好的模式(这种模式会提高训练数据上的性能,但不会提高测试数据上的性能)。

# 这里的架构是：
# 两个中间层,每层都有 16 个隐藏单元;第三层输出一个标量,预测当前评论的情感。
# 中间层使用 relu 作为激活函数,最后一层使用 sigmoid 激活以输出一个 0~1 范围内的概率值(表示样本的目标值等于 1 的可能性,即评论为正面的可能性)。
# relu (rectified linear unit,整流线性单元)函数将所有负值归零(见图 3-4),而 sigmoid 函数则将任意值“压缩”到 [0,1] 区间内,其输出值可以看作概率值。


from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# 最后,你需要选择损失函数和优化器。由于你面对的是一个二分类问题,网络输出是一个概率值(网络最后一层使用 sigmoid 激活函数,仅包含一个单元),那么最好使用 binary_crossentropy (二元交叉熵)损失。
# 这并不是唯一可行的选择,比如你还可以使用 mean_squared_error (均方误差)。
# 但对于输出概率值的模型,交叉熵(crossentropy)往往是最好的选择。
# 交叉熵是来自于信息论领域的概念,用于衡量概率分布之间的距离,在这个例子中就是真实分布与预测值之间的距离。
# 下面的步骤是用 rmsprop 优化器和 binary_crossentropy 损失函数来配置模型。注意,我们还在训练过程中监控精度。

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])


# 上述代码将优化器、损失函数和指标作为字符串传入,这是因为 rmsprop 、 binary_crossentropy 和 accuracy 都是 Keras 内置的一部分。
# 有时你可能希望配置自定义优化器的参数,或者传入自定义的损失函数或指标函数。
# 前者可通过向 optimizer 参数传入一个优化器类实例来实现,如下所示;

from keras import optimizers
# 配置优化器
model.compile(optimizer=optimizers.RMSprop(lr=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

from keras import losses
from keras import metrics

# 使用自定义的损失和指标
# 通过向 loss 和 metrics 参数传入函数对象来实现,自定义的损失函数或指标函数。
model.compile(optimizer=optimizers.RMSprop(lr=0.001),
              loss=losses.binary_crossentropy,
              metrics=[metrics.binary_accuracy])


# 为了在训练过程中监控模型在前所未见的数据上的精度,你需要将原始训练数据留出 10 000个样本作为验证集。

x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]


# 现在使用 512 个样本组成的小批量,将模型训练 20 个轮次(即对 x_train 和 y_train 两个张量中的所有样本进行 20 次迭代)。
# 与此同时,你还要监控在留出的 10 000 个样本上的损失和精度。你可以通过将验证数据传入 validation_data 参数来完成。

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))


# 在 CPU 上运行,每轮的时间不到 2 秒,训练过程将在 20 秒内结束。
# 每轮结束时会有短暂的停顿,因为模型要计算在验证集的 10 000 个样本上的损失和精度。
# 注意,调用 model.fit() 返回了一个 History 对象。这个对象有一个成员 history ,它是一个字典,包含训练过程中的所有数据。我们来看一下。
# >>> history_dict = history.history
# >>> history_dict.keys()
# 旧版本是：dict_keys(['val_acc', 'acc', 'val_loss', 'loss'])
# keras 2.2.4 版本是： dict_keys(['val_loss', 'val_binary_accuracy', 'loss', 'binary_accuracy'])


history_dict = history.history
history_dict.keys()

# 字典中包含 4 个条目,对应训练过程和验证过程中监控的指标。

# In[36]:
# 使用 Matplotlib 在同一张图上绘制训练损失和验证损失,以及训练精度和验证精度。
# 请注意,由于网络的随机初始化不同,你得到的结果可能会略有不同。

import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['LiSu'] # 步骤一（替换sans-serif字体）
plt.rcParams['axes.unicode_minus'] = False   # 步骤二（解决坐标轴负数的负号显示问题）

acc = history.history['binary_accuracy']

val_acc = history.history['val_binary_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

# 'bo' 表示蓝色圆点
plt.plot(epochs, loss, 'bo', label='训练损失(Training loss)')
# 'b' 表示蓝色实线
plt.plot(epochs, val_loss, 'b', label='验证损失(Validation loss)')
plt.title('训练损失和验证损失(Training and validation loss)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()


# In[38]:


plt.clf()   # 清空图像
acc_values = history_dict['binary_accuracy']
val_acc_values = history_dict['val_binary_accuracy']

plt.plot(epochs, acc, 'bo', label='训练精度(Training acc)')
plt.plot(epochs, val_acc, 'b', label='验证精度(Validation acc)')
plt.title('训练精度和验证精度(Training and validation accuracy)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()


# 训练损失每轮都在降低,训练精度每轮都在提升。这就是梯度下降优化的预期结果——你想要最小化的量随着每次迭代越来越小。
# 但验证损失和验证精度并非如此:它们似乎在第四轮达到最佳值。
# 这就是我们之前警告过的一种情况:模型在训练数据上的表现越来越好,但在前所未见的数据上不一定表现得越来越好。
# 准确地说,你看到的是过拟合(overfit):在第二轮之后,你对训练数据过度优化,最终学到的表示仅针对于训练数据,无法泛化到训练集之外的数据。
# 在这种情况下,为了防止过拟合,你可以在 3 轮之后停止训练。

# In[40]:

# 从头开始训练一个新的网络,训练 4 轮,然后在测试数据上评估模型。

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=4, batch_size=512)
results = model.evaluate(x_test, y_test)


# 最终结果如下所示。
# >>> results
# [0.2929924130630493, 0.88327999999999995]
# 这种相当简单的方法得到了 88% 的精度。利用最先进的方法,你应该能够得到接近 95% 的精度。



# 训练好网络之后,你希望将其用于实践。你可以用 predict 方法来得到评论为正面的可能性大小。
# >>> model.predict(x_test)
# array([[ 0.98006207]
#             [ 0.99758697]
#             [ 0.99975556]
#             ...,
#             [ 0.82167041]
#             [ 0.02885115]
#             [ 0.65371346]], dtype=float32)
# 如你所见,网络对某些样本的结果非常确信(大于等于 0.99,或小于等于 0.01),但对其他结果却不那么确信(0.6 或 0.4)。