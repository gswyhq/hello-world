#!/usr/bin/env python
# coding: utf-8

# 来源： https://github.com/fchollet/deep-learning-with-python-notebooks.gi
# In[1]:


import keras
keras.__version__ # 2.2.4


# 构建一个网络,将路透社新闻划分为 46 个互斥的主题。
# 因为有多个类别,所以这是多分类(multiclass classification)问题的一个例子。
# 因为每个数据点只能划分到一个类别,所以更具体地说,这是单标签、多分类(single-label, multiclass classification)问题的一个例子。
# 如果每个数据点可以划分到多个类别(主题),那它就是一个多标签、多分类(multilabel,multiclass classification)问题。


from keras.datasets import reuters

# 使用路透社数据集,它包含许多短新闻及其对应的主题,由路透社在 1986 年发布。
# 它是一个简单的、广泛使用的文本分类数据集。它包括 46 个不同的主题:某些主题的样本更多,但训练集中每个主题都有至少 10 个样本。

(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)


# 参数 num_words=10000 将数据限定为前 10 000 个最常出现的单词。我们有 8982 个训练样本和 2246 个测试样本。

# >>> len(train_data)
# 8982
# >>> len(test_data)
# 2246
#
# 每个样本都是一个整数列表(表示单词索引)。
# >>> train_data[10]
# [1, 245, 273, 207, 156, 53, 74, 160, 26, 14, 46, 296, 26, 39, 74, 2979,
# 3554, 14, 46, 4689, 4329, 86, 61, 3499, 4795, 14, 61, 451, 4329, 17, 12]


# 可以用下列代码将索引解码为单词。

word_index = reuters.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
# 注 意, 索 引 减 去 了 3, 因 为 0、1、2 是 为“padding”( 填 充 )、“start of sequence”(序列开始)、“unknown”(未知词)分别保留的索引
decoded_newswire = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])


# In[7]:


decoded_newswire


# 样本对应的标签是一个 0~45 范围内的整数,即话题索引编号。
# >>> train_labels[10]
# 3


# 准备数据
# 你可以使用与上一个例子相同的代码将数据向量化。

import numpy as np

def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

# 将训练数据向量化
x_train = vectorize_sequences(train_data)
# 将测试数据向量化
x_test = vectorize_sequences(test_data)


# 将标签向量化有两种方法:你可以将标签列表转换为整数张量,或者使用 one-hot 编码。
# one-hot 编码是分类数据广泛使用的一种格式,也叫分类编码(categorical encoding)。
# 在这个例子中,标签的 one-hot 编码就是将每个标签表示为全零向量,只有标签索引对应的元素为 1。
# 其代码实现如下。

def to_one_hot(labels, dimension=46):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1.
    return results

# 将训练标签向量化
one_hot_train_labels = to_one_hot(train_labels)
# 将测试标签向量化
one_hot_test_labels = to_one_hot(test_labels)


# Keras 内置方法可以实现`分类编码`这个操作

# In[11]:


from keras.utils.np_utils import to_categorical

one_hot_train_labels = to_categorical(train_labels)
one_hot_test_labels = to_categorical(test_labels)


# 构建网络
# 这个主题分类问题与前面的电影评论分类问题类似,两个例子都是试图对简短的文本片段进行分类。
# 但这个问题有一个新的约束条件:输出类别的数量从 2 个变为 46 个。
# 输出空间的维度要大得多。
# 对于前面用过的 Dense 层的堆叠,每层只能访问上一层输出的信息。
# 如果某一层丢失了与分类问题相关的一些信息,那么这些信息无法被后面的层找回,也就是说,每一层都可能成为信息瓶颈。
# 上一个例子使用了 16 维的中间层,但对这个例子来说 16 维空间可能太小了,无法学会区分 46 个不同的类别。
# 这种维度较小的层可能成为信息瓶颈,永久地丢失相关信息。
# 出于这个原因,下面将使用维度更大的层,包含 64 个单元。

# In[12]:


from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))

# 关于这个架构还应该注意另外两点。
# 网络的最后一层是大小为 46 的 Dense 层。
# 这意味着,对于每个输入样本,网络都会输出一个 46 维向量。
# 这个向量的每个元素(即每个维度)代表不同的输出类别。
# 最后一层使用了 softmax 激活。你在 MNIST 例子中见过这种用法。
# 网络将输出在 46个不同输出类别上的概率分布——对于每一个输入样本,网络都会输出一个 46 维向量,其中 output[i] 是样本属于第 i 个类别的概率。
# 46 个概率的总和为 1。对于这个例子,最好的损失函数是 categorical_crossentropy (分类交叉熵)。
# 它用于衡量两个概率分布之间的距离,这里两个概率分布分别是网络输出的概率分布和标签的真实分布。
# 通过将这两个分布的距离最小化,训练网络可使输出结果尽可能接近真实标签。


# 编译模型
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# 我们在训练数据中留出 1000 个样本作为验证集。

x_val = x_train[:1000]
partial_x_train = x_train[1000:]

y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]


# 现在开始训练网络,共 20 个轮次。


history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))


# 绘制损失曲线和精度曲线

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['LiSu'] # 步骤一（替换sans-serif字体）
plt.rcParams['axes.unicode_minus'] = False   # 步骤二（解决坐标轴负数的负号显示问题）

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'bo', label='训练损失(Training loss)')
plt.plot(epochs, val_loss, 'b', label='验证损失(Validation loss)')
plt.title('训练损失和验证损失(Training and validation loss)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()


# In[18]:


plt.clf()   # clear figure

acc = history.history['binary_accuracy']
val_acc = history.history['val_binary_accuracy']

plt.plot(epochs, acc, 'bo', label='训练精度(Training acc)')
plt.plot(epochs, val_acc, 'b', label='验证精度(Validation acc)')
plt.title('训练精度和验证精度(Training and validation accuracy)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()


# 网络在训练 8轮后开始过拟合。我们从头开始训练一个新网络,共 8 个轮次,然后在测试集上评估模型。

# In[27]:


model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(partial_x_train,
          partial_y_train,
          epochs=8,
          batch_size=512,
          validation_data=(x_val, y_val))
results = model.evaluate(x_test, one_hot_test_labels)


# 最终结果如下。
# >>> results
# [0.9565213431445807, 0.79697239536954589]
# 这种方法可以得到约 80% 的精度。对于平衡的二分类问题,完全随机的分类器能够得到
# 50% 的精度。但在这个例子中,完全随机的精度约为 19%,所以上述结果相当不错,至少和随
# 机的基准比起来还不错。
# >>> import copy
# >>> test_labels_copy = copy.copy(test_labels)
# >>> np.random.shuffle(test_labels_copy)
# >>> hits_array = np.array(test_labels) == np.array(test_labels_copy)
# >>> float(np.sum(hits_array)) / len(test_labels)
# 0.18655387355298308
# In[29]:


import copy

test_labels_copy = copy.copy(test_labels)
np.random.shuffle(test_labels_copy)
float(np.sum(np.array(test_labels) == np.array(test_labels_copy))) / len(test_labels)



# 你可以验证,模型实例的 predict 方法返回了在 46 个主题上的概率分布。我们对所有测
# 试数据生成主题预测。

# 在新数据上生成预测结果
predictions = model.predict(x_test)
# 2
# predictions 中的每个元素都是长度为 46 的向量。
# >>> predictions[0].shape
# (46,)
# 这个向量的所有元素总和为 1。
# 3
# >>> np.sum(predictions[0])
# 1.0
# 最大的元素就是预测类别,即概率最大的类别。
# >>> np.argmax(predictions[0])
# 4

# 前面提到了另一种编码标签的方法,就是将其转换为整数张量


y_train = np.array(train_labels)
y_test = np.array(test_labels)


# 对于这种编码方法,唯一需要改变的是损失函数的选择。使用的损失函数 categorical_crossentropy ,标签应该遵循分类编码。对于整数标签,你应该使用sparse_categorical_crossentropy 。

# In[36]:


model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['acc'])


# 这个新的损失函数在数学上与 categorical_crossentropy 完全相同,二者只是接口不同。

# 中间层维度足够大的重要性前面提到,最终输出是 46 维的,因此中间层的隐藏单元个数不应该比 46 小太多。现在来看一下,如果中间层的维度远远小于 46(比如 4 维),造成了信息瓶颈

# In[42]:


model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(4, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(partial_x_train,
          partial_y_train,
          epochs=20,
          batch_size=128,
          validation_data=(x_val, y_val))


# 现在网络的验证精度最大约为 71%,比前面下降了 8%。
# 导致这一下降的主要原因在于,你试图将大量信息(这些信息足够恢复 46 个类别的分割超平面)压缩到维度很小的中间空间。
# 网络能够将大部分必要信息塞入这个四维表示中,但并不是全部信息。

# 小结
#  如果要对 N 个类别的数据点进行分类,网络的最后一层应该是大小为 N 的 Dense 层。 对于单标签、多分类问题,网络的最后一层应该使用 softmax 激活,这样可以输出在 N个输出类别上的概率分布。
#  这种问题的损失函数几乎总是应该使用分类交叉熵。它将网络输出的概率分布与目标的真实分布之间的距离最小化。 处理多分类问题的标签有两种方法。
#  通 过 分 类 编 码( 也 叫 one-hot 编 码 ) 对 标 签 进 行 编 码, 然 后 使 用 categorical_crossentropy 作为损失函数。
#  将标签编码为整数,然后使用 sparse_categorical_crossentropy 损失函数。
#  如果你需要将数据划分到许多类别中,应该避免使用太小的中间层,以免在网络中造成信息瓶颈。
