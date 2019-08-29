#!/usr/bin/env python
# coding: utf-8

# 来源： https://github.com/fchollet/deep-learning-with-python-notebooks.gi
# In[1]:


import keras
keras.__version__ # '2.2.4'


# 即使用卷积神经网络对 MNIST 数字进行分类,这个任务我们在第 2 章用密集连接网络做过(当时的测试精度为 97.8%)。虽然本例中的卷积神经网络很简单,但其精度肯定会超过密集连接网络。
# 下列代码将会展示一个简单的卷积神经网络。它是 Conv2D 层和 MaxPooling2D 层的堆叠。


# 重要的是,卷积神经网络接收形状为 (image_height, image_width, image_channels)的输入张量(不包括批量维度)。
# 本例中设置卷积神经网络处理大小为 (28, 28, 1) 的输入张量,这正是 MNIST 图像的格式。我们向第一层传入参数 input_shape=(28, 28, 1) 来完成此设置。


# In[2]:


from keras import layers
from keras import models

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))


# 我们来看一下目前卷积神经网络的架构。
model.summary()


# 可以看到,每个 Conv2D 层和 MaxPooling2D 层的输出都是一个形状为 (height, width, channels) 的 3D 张量。
# 宽度和高度两个维度的尺寸通常会随着网络加深而变小。通道数量由传入 Conv2D 层的第一个参数所控制(32 或 64)。
# 下一步是将最后的输出张量[大小为 (3, 3, 64) ]输入到一个密集连接分类器网络中,即 Dense 层的堆叠,你已经很熟悉了。
# 这些分类器可以处理 1D 向量,而当前的输出是 3D 张量。首先,我们需要将 3D 输出展平为 1D,然后在上面添加几个 Dense 层。

# 在卷积神经网络上添加分类器
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))


# 我们将进行 10 类别分类,最后一层使用带 10 个输出的 softmax 激活。现在网络的架构如下。
model.summary()


# 如你所见,在进入两个 Dense 层之前,形状 (3, 3, 64) 的输出被展平为形状 (576,) 的向量。下面我们在 MNIST 数字图像上训练这个卷积神经网络。

# In[6]:


from keras.datasets import mnist
from keras.utils import to_categorical

# 将从`https://s3.amazonaws.com/img-datasets/mnist.npz`下载数据到： ~/.keras/datasets/mnist.npz
# 下载链接：https://pan.baidu.com/s/1jH6uFFC 密码: dw3d

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)


# In[7]:


model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5, batch_size=64)


# 我们在测试数据上对模型进行评估。

test_loss, test_acc = model.evaluate(test_images, test_labels)


# In[9]:


print(test_acc)

# 密集连接网络的测试精度为 97.8%,但这个简单卷积神经网络的测试精度达到了99.3%,我们将错误率降低了 68%(相对比例)。

# 密集连接层和卷积层的根本区别在于, Dense 层从输入特征空间中学到的是全局模式(比如对于 MNIST 数字,
# 全局模式就是涉及所有像素的模式),而卷积层学到的是局部模式,对于图像来说,学到的就是在输入图像的二维小窗口中发现的模式。

# 这个重要特性使卷积神经网络具有以下两个有趣的性质。
#  卷积神经网络学到的模式具有平移不变性(translation invariant)右下角学到某个模式之后,它可以在任何地方识别这个模式,比如左上角。
# 对于密集连接网络来说,如果模式出现在新的位置,它只能重新学习这个模式。
# 这使得卷积神经网络在处理图像时可以高效利用数据(因为视觉世界从根本上具有平移不变性),它只需要更少的训练样本就可以学到具有泛化能力的数据表示。

#  卷积神经网络可以学到模式的空间层次结构(spatial hierarchies of patterns)第一个卷积层将学习较小的局部模式(比如边缘)组成的更大的模式,以此类推。
# 这使得卷积神经网络可以有效地学习越来越复杂、越来越抽象的视觉概念(因为视觉世界从根本上具有空间层次结构)。

# 最大池化运算
# 在卷积神经网络示例中,你可能注意到,在每个 MaxPooling2D 层之后,特征图的尺寸都会减半。
# 例如,在第一个 MaxPooling2D 层之前,特征图的尺寸是 26×26,但最大池化运算将其减半为 13×13。
# 这就是最大池化的作用:对特征图进行下采样,与步进卷积类似。
# 最大池化是从输入特征图中提取窗口,并输出每个通道的最大值。
# 它的概念与卷积类似,但是最大池化使用硬编码的 max 张量运算对局部图块进行变换,而不是使用学到的线性变换(卷积核)。
# 最大池化与卷积的最大不同之处在于,最大池化通常使用 2×2 的窗口和步幅 2,其目的是将特征图下采样 2 倍。
# 与此相对的是,卷积通常使用 3×3 窗口和步幅 1。
# 为什么要用这种方式对特征图下采样?为什么不删除最大池化层,一直保留较大的特征图?

# 若删除池化层，这种架构有什么问题?有如下两点问题。
#
#  这种架构不利于学习特征的空间层级结构。第三层的 3×3 窗口中只包含初始输入的
# 7×7 窗口中所包含的信息。卷积神经网络学到的高级模式相对于初始输入来说仍然很小,
# 这可能不足以学会对数字进行分类(你可以试试仅通过 7 像素×7 像素的窗口观察图像
# 来识别其中的数字)。我们需要让最后一个卷积层的特征包含输入的整体信息。
#
#  最后一层的特征图对每个样本共有 22×22×64=30 976 个元素。这太多了。如果你将其
# 展平并在上面添加一个大小为 512 的 Dense 层,那一层将会有 1580 万个参数。这对于
# 这样一个小模型来说太多了,会导致严重的过拟合。
#
# 简而言之,使用下采样的原因,一是减少需要处理的特征图的元素个数,二是通过让连续
# 卷积层的观察窗口越来越大(即窗口覆盖原始输入的比例越来越大),从而引入空间过滤器的层
# 级结构。
#
# 注意,最大池化不是实现这种下采样的唯一方法。你已经知道,还可以在前一个卷积层中
# 使用步幅来实现。此外,你还可以使用平均池化来代替最大池化,其方法是将每个局部输入图
# 块变换为取该图块各通道的平均值,而不是最大值。但最大池化的效果往往比这些替代方法更好。
#
# 简而言之,原因在于特征中往往编码了某种模式或概念在特征图的不同位置是否存在(因此得
# 名特征图),而观察不同特征的最大值而不是平均值能够给出更多的信息。因此,最合理的子采
# 样策略是首先生成密集的特征图(通过无步进的卷积),然后观察特征每个小图块上的最大激活,
# 而不是查看输入的稀疏窗口(通过步进卷积)或对输入图块取平均,因为后两种方法可能导致
# 错过或淡化特征是否存在的信息。
