#!/usr/bin/env python
# coding: utf-8
# 来源： https://github.com/fchollet/deep-learning-with-python-notebooks.git

# In[1]:
import keras
keras.__version__  # '2.2.4'
# 加载 Keras 中的 手写数字分类MNIST 数据集
from keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
# train_images 和 train_labels 组成了训练集(training set),模型将从这些数据中进行学习。
# 然后在测试集(test set,即 test_images 和 test_labels )上对模型进行测试。
# 图像被编码为 Numpy 数组,而标签是数字数组,取值范围为 0~9。图像和标签一一对应。
# 我们来看一下训练数据:
# In[3]:
train_images.shape
# In[4]:
len(train_labels)
# In[5]:
train_labels
# 下面是测试数据:
# In[6]:
test_images.shape
# In[7]:
len(test_labels)
# In[8]:
test_labels

# 接下来的工作流程如下:首先,将训练数据( train_images 和 train_labels )输入神经网络;
# 其次,网络学习将图像和标签关联在一起;最后,网络对 test_images 生成预测,
# 而我们将验证这些预测与 test_labels 中的标签是否匹配。


from keras import models
from keras import layers

# 构建网络
# 这个网络包含两个 Dense 层,每层都对输入数据进行一些简单的张量运算,
# 这些运算都包含权重张量。权重张量是该层的属性,里面保存了网络所学到的知识 (knowledge)。

network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))

# 本例中的网络包含 2 个 Dense 层,它们是密集连接(也叫全连接)的神经层。
# 第二层(也是最后一层)是一个 10 路 softmax 层,它将返回一个由 10 个概率值(总和为 1)组成的数组。
# 每个概率值表示当前数字图像属于 10 个数字类别中某一个的概率。

# 要想训练网络,我们还需要选择编译(compile)步骤的三个参数。
#  损失函数(loss function):网络如何衡量在训练数据上的性能,即网络如何朝着正确的方向前进。
#  优化器(optimizer):基于训练数据和损失函数来更新网络的机制。
#  在训练和测试过程中需要监控的指标(metric):本例只关心精度,即正确分类的图像所占的比例。

# 编译网络
# categorical_crossentropy 是损失函数,是用于学习权重张量的反馈信号,在训练阶段应使它最小化。
# 减小损失是通过小批量随机梯度下降来实现的。
# 梯度下降的具体方法由第一个参数给定,即 rmsprop 优化器。

network.compile(optimizer='rmsprop',
                            loss='categorical_crossentropy',
                            metrics=['accuracy'])
#

# 输入图像保存在 float32 格式的 Numpy 张量中,形状分别为 (60000,28*28) (训练数据)和 (10000, 28*28) (测试数据)。
# 在开始训练之前,我们将对数据进行预处理,将其变换为网络要求的形状,并缩放到所有值都在 [0, 1] 区间。
# 比如,之前训练图像保存在一个 uint8 类型的数组中,其形状为 (60000, 28, 28) ,取值区间为 [0, 255] 。
# 我们需要将其变换为一个 float32 数组,其形 状为 (60000, 28 * 28) ,取值范围为 0~1。
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255
# 我们还需要对标签进行分类编码
# In[12]:
from keras.utils import to_categorical
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
# 现在我们准备开始训练网络,在 Keras 中这一步是通过调用网络的 fit 方法来完成的——
# 我们在训练数据上拟合(fit)模型。
# In[13]:
network.fit(train_images, train_labels, epochs=5, batch_size=128)
# 网络开始在训练数据上进行迭代(每个小批量包含128 个样本),共迭代 5 次[在所有训练数据上迭代一次叫作一个轮次(epoch)]。
# 在每次迭代过程中,网络会计算批量损失相对于权重的梯度,并相应地更新权重。
# 5 轮之后,网络进行了2345 次梯度更新(每轮 469 次; 60000/128 ＝ 468.75≈ 469),网络损失值将变得足够小,使得网络能够以很高的精度对手写数字进行分类。


# 训练过程中显示了两个数字:一个是网络在训练数据上的损失( loss ),另一个是网络在训练数据上的精度( acc )。
# 我们很快就在训练数据上达到了 0.989(98.9%)的精度。现在我们来检查一下模型在测试集上的性能。
# In[14]:
test_loss, test_acc = network.evaluate(test_images, test_labels)
# In[15]:
print('test_acc:', test_acc)
# 测试集精度为 97.8%,比训练集精度低不少。训练精度和测试精度之间的这种差距是过拟合(overfit)造成的。
