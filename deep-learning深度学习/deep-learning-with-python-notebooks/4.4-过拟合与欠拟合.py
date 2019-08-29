#!/usr/bin/env python
# coding: utf-8

# 来源： https://github.com/fchollet/deep-learning-with-python-notebooks.gi
# In[1]:


import keras
keras.__version__

# 过拟合与欠拟合
# 在上一章的三个例子(预测电影评论、主题分类和房价回归)中,模型在留出验证数据上的性能总是在几轮后达到最高点,然后开始下降。
# 也就是说,模型很快就在训练数据上开始过拟合。过拟合存在于所有机器学习问题中。学会如何处理过拟合对掌握机器学习至关重要。
# 机器学习的根本问题是优化和泛化之间的对立。
# 优化(optimization)是指调节模型以在训练数据上得到最佳性能(即机器学习中的学习),而泛化(generalization)是指训练好的模型在前所未见的数据上的性能好坏。
# 机器学习的目的当然是得到良好的泛化,但你无法控制泛化,只能基于训练数据调节模型。训练开始时,优化和泛化是相关的:训练数据上的损失越小,测试数据上的损失也越小。
# 这时的模型是欠拟合(underfit)的,即仍有改进的空间,网络还没有对训练数据中所有相关模式建模。
# 但在训练数据上迭代一定次数之后,泛化不再提高,验证指标先是不变,然后开始变差,即模型开始过拟合。
# 这时模型开始学习仅和训练数据有关的模式,但这种模式对新数据来说是错误的或无关紧要的。
# 为了防止模型从训练数据中学到错误或无关紧要的模式,最优解决方法是获取更多的训练数据。
# 模型的训练数据越多,泛化能力自然也越好。如果无法获取更多数据,次优解决方法是调节模型允许存储的信息量,或对模型允许存储的信息加以约束。
# 如果一个网络只能记住几个模式,那么优化过程会迫使模型集中学习最重要的模式,这样更可能得到良好的泛化。
# 这种降低过拟合的方法叫作正则化(regularization)。
# 我们先介绍几种最常见的正则化方法,然后将其应用于实践中,以改进 3.4 节的电影分类模型。
#
# 减小网络大小
# 防止过拟合的最简单的方法就是减小模型大小,即减少模型中可学习参数的个数(这由层数和每层的单元个数决定)。
# 在深度学习中,模型中可学习参数的个数通常被称为模型的容量(capacity)。
# 直观上来看,参数更多的模型拥有更大的记忆容量(memorization capacity),因此能够在训练样本和目标之间轻松地学会完美的字典式映射,这种映射没有任何泛化能力。
# 例如,拥有 500 000 个二进制参数的模型,能够轻松学会 MNIST 训练集中所有数字对应的类别——我们只需让 50 000 个数字每个都对应 10 个二进制参数。
# 但这种模型对于新数字样本的分类毫无用处。始终牢记:深度学习模型通常都很擅长拟合训练数据,但真正的挑战在于泛化,而不是拟合。
# 与此相反,如果网络的记忆资源有限,则无法轻松学会这种映射。
# 因此,为了让损失最小化,网络必须学会对目标具有很强预测能力的压缩表示,这也正是我们感兴趣的数据表示。
# 同时请记住,你使用的模型应该具有足够多的参数,以防欠拟合,即模型应避免记忆资源不足。
# 在容量过大与容量不足之间要找到一个折中。
# 不幸的是,没有一个魔法公式能够确定最佳层数或每层的最佳大小。
# 你必须评估一系列不同的网络架构(当然是在验证集上评估,而不是在测试集上),以便为数据找到最佳的模型大小。
# 要找到合适的模型大小,一般的工作流程是开始时选择相对较少的层和参数,然后逐渐增加层的大小或增加新层,直到这种增加对验证损失的影响变得很小。



# In[2]:


from keras.datasets import imdb
import numpy as np

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

def vectorize_sequences(sequences, dimension=10000):
    # Create an all-zero matrix of shape (len(sequences), dimension)
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.  # set specific indices of results[i] to 1s
    return results

# Our vectorized training data
x_train = vectorize_sequences(train_data)
# Our vectorized test data
x_test = vectorize_sequences(test_data)
# Our vectorized labels
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')



# In[4]:
# 我们在电影评论分类的网络上试一下。原始网络如下所示。

from keras import models
from keras import layers

original_model = models.Sequential()
original_model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
original_model.add(layers.Dense(16, activation='relu'))
original_model.add(layers.Dense(1, activation='sigmoid'))

original_model.compile(optimizer='rmsprop',
                       loss='binary_crossentropy',
                       metrics=['acc'])


# 现在我们尝试用下面这个更小的网络来替换它。
# 容量更小的模型
# In[5]:
smaller_model = models.Sequential()
smaller_model.add(layers.Dense(4, activation='relu', input_shape=(10000,)))
smaller_model.add(layers.Dense(4, activation='relu'))
smaller_model.add(layers.Dense(1, activation='sigmoid'))

smaller_model.compile(optimizer='rmsprop',
                      loss='binary_crossentropy',
                      metrics=['acc'])


# 比较了原始网络与更小网络的验证损失。圆点是更小网络的验证损失值,十字是原始网络的验证损失值(请记住,更小的验证损失对应更好的模型)。
# In[6]:


original_hist = original_model.fit(x_train, y_train,
                                   epochs=20,
                                   batch_size=512,
                                   validation_data=(x_test, y_test))


# In[7]:


smaller_model_hist = smaller_model.fit(x_train, y_train,
                                       epochs=20,
                                       batch_size=512,
                                       validation_data=(x_test, y_test))


# In[8]:


epochs = range(1, 21)
original_val_loss = original_hist.history['val_loss']
smaller_model_val_loss = smaller_model_hist.history['val_loss']


# In[9]:


import matplotlib.pyplot as plt

# b+ is for "blue cross"
plt.plot(epochs, original_val_loss, 'b+', label='Original model')
# "bo" is for "blue dot"
plt.plot(epochs, smaller_model_val_loss, 'bo', label='Smaller model')
plt.xlabel('Epochs')
plt.ylabel('Validation loss')
plt.legend()

plt.show()


# 更小的网络开始过拟合的时间要晚于参考网络(前者 6 轮后开始过拟合,而后者 4 轮后开始),而且开始过拟合之后,它的性能变差的速度也更慢。
# 现在,我们再向这个基准中添加一个容量更大的网络(容量远大于问题所需)。

# In[11]:


bigger_model = models.Sequential()
bigger_model.add(layers.Dense(512, activation='relu', input_shape=(10000,)))
bigger_model.add(layers.Dense(512, activation='relu'))
bigger_model.add(layers.Dense(1, activation='sigmoid'))

bigger_model.compile(optimizer='rmsprop',
                     loss='binary_crossentropy',
                     metrics=['acc'])


# In[12]:


bigger_model_hist = bigger_model.fit(x_train, y_train,
                                     epochs=20,
                                     batch_size=512,
                                     validation_data=(x_test, y_test))


# 显示了更大的网络与参考网络的性能对比。圆点是更大网络的验证损失值,十字是原始网络的验证损失值。

# In[26]:


bigger_model_val_loss = bigger_model_hist.history['val_loss']

plt.plot(epochs, original_val_loss, 'b+', label='Original model')
plt.plot(epochs, bigger_model_val_loss, 'bo', label='Bigger model')
plt.xlabel('Epochs')
plt.ylabel('Validation loss')
plt.legend()

plt.show()


# 更大的网络只过了一轮就开始过拟合,过拟合也更严重。其验证损失的波动也更大。
# In[28]:


original_train_loss = original_hist.history['loss']
bigger_model_train_loss = bigger_model_hist.history['loss']

plt.plot(epochs, original_train_loss, 'b+', label='Original model')
plt.plot(epochs, bigger_model_train_loss, 'bo', label='Bigger model')
plt.xlabel('Epochs')
plt.ylabel('Training loss')
plt.legend()

plt.show()


# 更大网络的训练损失很快就接近于零。
# 网络的容量越大,它拟合训练数据(即得到很小的训练损失)的速度就越快,但也更容易过拟合(导致训练损失和验证损失有很大差异)。

# 添加权重正则化
# 你可能知道奥卡姆剃刀(Occam’s razor)原理:如果一件事情有两种解释,那么最可能正确的解释就是最简单的那个,即假设更少的那个。
# 这个原理也适用于神经网络学到的模型:给定一些训练数据和一种网络架构,很多组权重值(即很多模型)都可以解释这些数据。
# 简单模型比复杂模型更不容易过拟合。
#
# 这里的简单模型(simple model)是指参数值分布的熵更小的模型(或参数更少的模型,比如上一节的例子)。
# 因此,一种常见的降低过拟合的方法就是强制让模型权重只能取较小的值,从而限制模型的复杂度,这使得权重值的分布更加规则(regular)。
# 这种方法叫作权重正则化(weight regularization),其实现方法是向网络损失函数中添加与较大权重值相关的成本(cost)。这个成本有两种形式。
#  L1 正则化(L1 regularization):添加的成本与权重系数的绝对值[权重的 L1 范数(norm)]成正比。
#  L2 正则化(L2 regularization):添加的成本与权重系数的平方(权重的 L2 范数)成正比。
# 神经网络的 L2 正则化也叫权重衰减(weight decay)。不要被不同的名称搞混,权重衰减与 L2 正则化在数学上是完全相同的。
# 在 Keras 中, 添 加 权 重 正 则 化 的 方 法 是 向 层 传 递权 重 正 则 化 项 实 例(weight regularizer instance)作为关键字参数。
# 下列代码将向电影评论分类网络中添加 L2 权重正则化。

# In[17]:


from keras import regularizers

l2_model = models.Sequential()
l2_model.add(layers.Dense(16, kernel_regularizer=regularizers.l2(0.001),
                          activation='relu', input_shape=(10000,)))
l2_model.add(layers.Dense(16, kernel_regularizer=regularizers.l2(0.001),
                          activation='relu'))
l2_model.add(layers.Dense(1, activation='sigmoid'))


# In[18]:


l2_model.compile(optimizer='rmsprop',
                 loss='binary_crossentropy',
                 metrics=['acc'])


# l2(0.001) 的意思是该层权重矩阵的每个系数都会使网络总损失增加 0.001 * weight_coefficient_value 。注意,由于这个惩罚项只在训练时添加,所以这个网络的训练损失会比测试损失大很多。

# L2 正则化惩罚的影响。如你所见,即使两个模型的参数个数相同,具有 L2正则化的模型(圆点)比参考模型(十字)更不容易过拟合。


# In[19]:


l2_model_hist = l2_model.fit(x_train, y_train,
                             epochs=20,
                             batch_size=512,
                             validation_data=(x_test, y_test))


# In[30]:


l2_model_val_loss = l2_model_hist.history['val_loss']

plt.plot(epochs, original_val_loss, 'b+', label='Original model')
plt.plot(epochs, l2_model_val_loss, 'bo', label='L2-regularized model')
plt.xlabel('Epochs')
plt.ylabel('Validation loss')
plt.legend()

plt.show()


# 你还可以用 Keras 中以下这些权重正则化项来代替 L2 正则化。

# In[ ]:


from keras import regularizers

# L1 正则化
regularizers.l1(0.001)

# 同时做 L1 和 L2 正则化
regularizers.l1_l2(l1=0.001, l2=0.001)


# 添加 dropout 正则化
# dropout 是神经网络最有效也最常用的正则化方法之一,它是由多伦多大学的 Geoffrey Hinton和他的学生开发的。
# 对某一层使用 dropout,就是在训练过程中随机将该层的一些输出特征舍弃(设置为 0)。
# 假设在训练过程中,某一层对给定输入样本的返回值应该是向量 [0.2, 0.5,1.3, 0.8, 1.1] 。
# 使用 dropout 后,这个向量会有几个随机的元素变成 0,比如 [0, 0.5,1.3, 0, 1.1] 。
# dropout 比率(dropout rate)是被设为 0 的特征所占的比例,通常在 0.2~0.5范围内。
# 测试时没有单元被舍弃,而该层的输出值需要按 dropout 比率缩小,因为这时比训练时有更多的单元被激活,需要加以平衡。
# 假 设 有 一 个 包 含 某 层 输 出 的 Numpy 矩 阵 layer_output , 其 形 状 为 (batch_size,features) 。
# 训练时,我们随机将矩阵中一部分值设为 0。
# In[ ]:

layer_output = None
model =None

# 训 练 时, 舍 弃 50%的输出单元
layer_output *= np.random.randint(0, high=2, size=layer_output.shape)


# 测试时,我们将输出按 dropout 比率缩小。这里我们乘以 0.5(因为前面舍弃了一半的单元)。
# 测试时
layer_output *= 0.5


# 注意,为了实现这一过程,还可以让两个运算都在训练时进行,而测试时输出保持不变。这通常也是实践中的实现方式
# In[ ]:


# 训练时:
layer_output *= np.random.randint(0, high=2, size=layer_output.shape)
# 注意,是成比例放大而不是成比例缩小
layer_output /= 0.5


# 在每个样本中随机删除不同的部分神经元。” a 其核心思想是在层的输出值中引入噪声,打破不显著的偶然模式。
# 如果没有噪声的话,网络将会记住这些偶然模式。在 Keras 中,你可以通过 Dropout 层向网络中引入 dropout,dropout 将被应用于前面一层的输出。


model.add(layers.Dropout(0.5))


# 我们向 IMDB 网络中添加两个 Dropout 层,来看一下它们降低过拟合的效果。

# In[22]:


dpt_model = models.Sequential()
dpt_model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
dpt_model.add(layers.Dropout(0.5))
dpt_model.add(layers.Dense(16, activation='relu'))
dpt_model.add(layers.Dropout(0.5))
dpt_model.add(layers.Dense(1, activation='sigmoid'))

dpt_model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['acc'])


# In[23]:


dpt_model_hist = dpt_model.fit(x_train, y_train,
                               epochs=20,
                               batch_size=512,
                               validation_data=(x_test, y_test))


# 我们再次看到,这种方法的性能相比参考网络有明显提高。

# In[32]:


dpt_model_val_loss = dpt_model_hist.history['val_loss']

plt.plot(epochs, original_val_loss, 'b+', label='Original model')
plt.plot(epochs, dpt_model_val_loss, 'bo', label='Dropout-regularized model')
plt.xlabel('Epochs')
plt.ylabel('Validation loss')
plt.legend()

plt.show()


# 总结一下,防止神经网络过拟合的常用方法包括:
#  获取更多的训练数据
#  减小网络容量
#  添加权重正则化
#  添加 dropout