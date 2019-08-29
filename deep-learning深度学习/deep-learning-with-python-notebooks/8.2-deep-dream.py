#!/usr/bin/env python
# coding: utf-8

# 来源： https://github.com/fchollet/deep-learning-with-python-notebooks.gi
# In[1]:


import keras
keras.__version__


# DeepDream 算法与卷积神经网络过滤器可视化技术几乎相同,都是反向运行
# 一个卷积神经网络:对卷积神经网络的输入做梯度上升,以便将卷积神经网络靠顶部的某一层
# 的某个过滤器激活最大化。DeepDream 使用了相同的想法,但有以下这几个简单的区别。
#  使用 DeepDream,我们尝试将所有层的激活最大化,而不是将某一层的激活最大化,因此需要同时将大量特征的可视化混合在一起。
#  不是从空白的、略微带有噪声的输入开始,而是从现有的图像开始,因此所产生的效果能够抓住已经存在的视觉模式,并以某种艺术性的方式将图像元素扭曲。
#  输入图像是在不同的尺度上[叫作八度 (octave)]进行处理的,这可以提高可视化的质量。我们来生成一些 DeepDream 图像。

# 用 Keras 实现 DeepDream
# 我们将从一个在 ImageNet 上预训练的卷积神经网络开始。Keras 中有许多这样的卷积神经
# 网络:VGG16、VGG19、Xception、ResNet50 等。我们可以用其中任何一个来实现 DeepDream,
# 但我们选择的卷积神经网络会影响可视化的效果,因为不同的卷积神经网络架构会学到不同的
# 特征。最初发布的 DeepDream 中使用的卷积神经网络是一个 Inception 模型,在实践中,人们已
# 经知道 Inception 能够生成漂亮的 DeepDream 图像,所以我们将使用 Keras 内置的 Inception V3
# 模型。

# In[2]:


from keras.applications import inception_v3
from keras import backend as K

# 我们不需要训练模型,所以这个命令会禁用所有与训练有关的操作
K.set_learning_phase(0)

# 构建不包括全连接层的 Inception V3网络。使用预训练的 ImageNet 权重来加载模型
model = inception_v3.InceptionV3(weights='imagenet',
                                 include_top=False)


# 接下来,我们要计算损失(loss),即在梯度上升过程中需要最大化的量。在第 5 章的过滤
# 器可视化中,我们试图将某一层的某个过滤器的值最大化。这里,我们要将多个层的所有过滤
# 器的激活同时最大化。具体来说,就是对一组靠近顶部的层激活的 L2 范数进行加权求和,然
# 后将其最大化。选择哪些层(以及它们对最终损失的贡献)对生成的可视化结果具有很大影响,
# 所以我们希望让这些参数变得易于配置。更靠近底部的层生成的是几何图案,而更靠近顶部的
# 层生成的则是从中能够看出某些 ImageNet 类别(比如鸟或狗)的图案。我们将随意选择 4 层的
# 配置,但你以后一定要探索多个不同的配置。

# In[3]:


# 这个字典将层的名称映射为一个系数,这个系数定量表示该层激活对你要最大化的损失的贡献大小。
# 注意,层的名称硬编码在内置的 Inception V3 应用中。可以使用 model.summary() 列出所有层的名称
layer_contributions = {
    'mixed2': 0.2,
    'mixed3': 3.,
    'mixed4': 2.,
    'mixed5': 1.5,
}


# 接下来,我们来定义一个包含损失的张量,损失就是上面层激活的 L2 范数的加权求和。


# 创建一个字典,将层的名称映射为层的实例
layer_dict = dict([(layer.name, layer) for layer in model.layers])

# 在定义损失时将层的贡献添加到这个标量变量中
loss = K.variable(0.)
for layer_name in layer_contributions:
    # 将该层特征的 L2 范数添加到 loss 中。
    coeff = layer_contributions[layer_name]
    # 获取层的输出
    activation = layer_dict[layer_name].output

    # 为了避免出现边界伪影,损失中仅包含非边界的像素
    scaling = K.prod(K.cast(K.shape(activation), 'float32'))
    loss += coeff * K.sum(K.square(activation[:, 2: -2, 2: -2, :])) / scaling


# 下面来设置梯度上升过程。

# 这个张量用于保存生成的图像,即梦境图像
dream = model.input

# 计算损失相对于梦境图像的梯度
grads = K.gradients(loss, dream)[0]

# 将梯度标准化(重要技巧)
grads /= K.maximum(K.mean(K.abs(grads)), 1e-7)

# 给定一张输出图像,设置一个 Keras 函数来获取损失值和梯度值
outputs = [loss, grads]
fetch_loss_and_grads = K.function([dream], outputs)

def eval_loss_and_grads(x):
    outs = fetch_loss_and_grads([x])
    loss_value = outs[0]
    grad_values = outs[1]
    return loss_value, grad_values

def gradient_ascent(x, iterations, step, max_loss=None):
    # 这个函数运行 iterations次梯度上升
    for i in range(iterations):
        loss_value, grad_values = eval_loss_and_grads(x)
        if max_loss is not None and loss_value > max_loss:
            break
        print('...Loss value at', i, ':', loss_value)
        x += step * grad_values
    return x


# 最后就是实际的 DeepDream 算法。首先,我们来定义一个列表,里面包含的是处理图像的尺度(也叫八度)。
# 每个连续的尺度都是前一个的 1.4 倍(放大 40%),即首先处理小图像,然后逐渐增大图像尺寸

# ![deep dream process](https://s3.amazonaws.com/book.keras.io/img/ch8/deepdream_process.png)

# 对于每个连续的尺度,从最小到最大,我们都需要在当前尺度运行梯度上升,以便将之前定义的损失最大化。
# 每次运行完梯度上升之后,将得到的图像放大 40%。在每次连续的放大之后(图像会变得模糊或像素化),为避免丢失大量图像细节,
# 我们可以使用一个简单的技巧:每次放大之后,将丢失的细节重新注入到图像中。
# 这种方法是可行的,因为我们知道原始图像放大到这个尺寸应该是什么样子。
# 给定一个较小的图像尺寸 S 和一个较大的图像尺寸 L,你可以计算将原始图像大小调整为 L 与将原始图像大小调整为 S 之间的区别,
# 这个区别可以定量描述从 S 到 L 的细节损失。

# 注意,上述代码使用了下面这些简单的 Numpy 辅助函数,其功能从名称中就可以看出来。它们都需要安装 SciPy。

# In[7]:


import scipy
from keras.preprocessing import image

def resize_img(img, size):
    img = np.copy(img)
    factors = (1,
               float(size[0]) / img.shape[1],
               float(size[1]) / img.shape[2],
               1)
    return scipy.ndimage.zoom(img, factors, order=1)


def save_img(img, fname):
    pil_img = deprocess_image(np.copy(img))
    scipy.misc.imsave(fname, pil_img)


def preprocess_image(image_path):
    # 通用函数,用于打开图像、改变图像大小以及将图像格式转换为 Inception V3 模型能够处理的张量
    img = image.load_img(image_path)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = inception_v3.preprocess_input(img)
    return img


def deprocess_image(x):
    # 通用函数,将一个张量转换为有效图像
    if K.image_data_format() == 'channels_first':
        x = x.reshape((3, x.shape[2], x.shape[3]))
        x = x.transpose((1, 2, 0))
    else:
        # 对 inception_v3.preprocess_input所做的预处理进行反向操作
        x = x.reshape((x.shape[1], x.shape[2], 3))
    x /= 2.
    x += 0.5
    x *= 255.
    x = np.clip(x, 0, 255).astype('uint8')
    return x


# In[9]:


import numpy as np

# 改变这些超参数,可以得到新的效果

step = 0.01  # 梯度上升的步长
num_octave = 3  # 运行梯度上升的尺度个数
octave_scale = 1.4  # 两个尺度之间的大小比例
iterations = 20  # 在每个尺度上运行梯度上升的步数

# 如果损失增大到大于 10,我们要中断梯度上升过程,以避免得到丑陋的伪影
max_loss = 10.

# 将这个变量修改为你要使用的图像的路径
base_image_path = '/home/gswyhq/data/original_photo_deep_dream.jpg'

# 将基础图像加载成一个 Numpy 数组
img = preprocess_image(base_image_path)

# 准备一个由形状元组组成的列表,它定义了运行梯度上升的不同尺度
original_shape = img.shape[1:3]
successive_shapes = [original_shape]
for i in range(1, num_octave):
    shape = tuple([int(dim / (octave_scale ** i)) for dim in original_shape])
    successive_shapes.append(shape)

# 将形状列表反转,变为升序
successive_shapes = successive_shapes[::-1]

# 将图像 Numpy 数组的大小缩放到最小尺寸
original_img = np.copy(img)
shrunk_original_img = resize_img(img, successive_shapes[0])

for shape in successive_shapes:
    print('Processing image shape', shape)
    # 将梦境图像放大
    img = resize_img(img, shape)
    # 运行梯度上升,改变梦境图像
    img = gradient_ascent(img,
                          iterations=iterations,
                          step=step,
                          max_loss=max_loss)
    # 将原始图像的较小版本放大,它会变得像素化
    upscaled_shrunk_original_img = resize_img(shrunk_original_img, shape)
    # 在这个尺寸上计算原始图像的高质量版本
    same_size_original = resize_img(original_img, shape)
    # 二者的差别就是在放大过程中丢失的细节
    lost_detail = same_size_original - upscaled_shrunk_original_img

    # 将丢失的细节重新注入到梦境图像中
    img += lost_detail
    shrunk_original_img = resize_img(original_img, shape)
    save_img(img, fname='dream_at_scale_' + str(shape) + '.png')

save_img(img, fname='final_dream.png')


# In[14]:


from matplotlib import pyplot as plt

plt.imshow(deprocess_image(np.copy(img)))
plt.show()

# 注意 因为原始 Inception V3 网络训练识别尺寸为 299×299 的图像中的概念,而上述过程中将
# 图像尺寸减小很多,所以 DeepDream 实现在尺寸介于 300×300 和 400×400 之间的图像
# 上能够得到更好的结果。但不管怎样,你都可以在任何尺寸和任何比例的图像上运行同
# 样的代码。

# DeepDream 的过程是反向运行一个卷积神经网络,基于网络学到的表示来生成输入。
#  得到的结果是很有趣的,有些类似于通过迷幻剂扰乱视觉皮层而诱发的视觉伪影。
#  注意,这个过程并不局限于图像模型,甚至并不局限于卷积神经网络。它可以应用于语音、音乐等更多内容。
