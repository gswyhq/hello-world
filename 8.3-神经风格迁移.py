#!/usr/bin/env python
# coding: utf-8

# 来源： https://github.com/fchollet/deep-learning-with-python-notebooks.gi
# In[1]:


import keras
keras.__version__


# 神经风格迁移
# 除 DeepDream 之外,深度学习驱动图像修改的另一项重大进展是神经风格迁移(neural
# style transfer),它由 Leon Gatys 等人于 2015 年夏天提出。 a 自首次提出以来,神经风格迁移算
# 法已经做了许多改进,并衍生出许多变体,而且还成功转化成许多智能手机图片应用。为了简单起见,本节将重点介绍原始论文中描述的方法。
# 神经风格迁移是指将参考图像的风格应用于目标图像,同时保留目标图像的内容。
# 
# 在当前语境下,风格(style)是指图像中不同空间尺度的纹理、颜色和视觉图案,内容
# (content)是指图像的高级宏观结构。举个例子,在图 8-7 中(用到的参考图像是文森特 • 梵高
# 的《星夜》),蓝黄色圆形笔划被看作风格,而 Tübingen(图宾根)照片中的建筑则被看作内容。
# 风格迁移这一想法与纹理生成的想法密切相关,在 2015 年开发出神经风格迁移之前,这一
# 想法就已经在图像处理领域有着悠久的历史。但事实证明,与之前经典的计算机视觉技术实现
# 相比,基于深度学习的风格迁移实现得到的结果是无与伦比的,并且还在计算机视觉的创造性
# 应用中引发了惊人的复兴。
# 实现风格迁移背后的关键概念与所有深度学习算法的核心思想是一样的:定义一个损失函
# 数来指定想要实现的目标,然后将这个损失最小化。你知道想要实现的目标是什么,就是保存
# 原始图像的内容,同时采用参考图像的风格。如果我们能够在数学上给出内容和风格的定义,
# 那么就有一个适当的损失函数(如下所示),我们将对其进行最小化。

# ```
# loss = distance(style(reference_image) - style(generated_image)) +
#        distance(content(original_image) - content(generated_image))
# ```

# 这里的 distance 是一个范数函数,比如 L2 范数; content 是一个函数,输入一张图
# 像,并计算出其内容的表示; style 是一个函数,输入一张图像,并计算出其风格的表示。将
# 这 个 损 失 最 小 化, 会 使 得 style(generated_image) 接 近 于 style(reference_image) 、
# content(generated_image) 接近于 content(generated_image) ,从而实现我们定义的
# 风格迁移。
# Gatys 等人发现了一个很重要的观察结果,就是深度卷积神经网络能够从数学上定义 style
# 和 content 两个函数。

# ## 内容损失
# 如你所知,网络更靠底部的层激活包含关于图像的局部信息,而更靠近顶部的层则包含更
# 加全局、更加抽象的信息。卷积神经网络不同层的激活用另一种方式提供了图像内容在不同空
# 间尺度上的分解。因此,图像的内容是更加全局和抽象的,我们认为它能够被卷积神经网络更
# 靠顶部的层的表示所捕捉到。
#
# 因此,内容损失的一个很好的候选者就是两个激活之间的 L2 范数,一个激活是预训练的卷
# 积神经网络更靠顶部的某层在目标图像上计算得到的激活,另一个激活是同一层在生成图像上
# 计算得到的激活。这可以保证,在更靠顶部的层看来,生成图像与原始目标图像看起来很相似。
# 假设卷积神经网络更靠顶部的层看到的就是输入图像的内容,那么这种方法可以保存图像内容。

# ## 风格损失
# 内容损失只使用了一个更靠顶部的层,但 Gatys 等人定义的风格损失则使用了卷积神经网
# 络的多个层。我们想要捉到卷积神经网络在风格参考图像的所有空间尺度上提取的外观,而不
# 仅仅是在单一尺度上。对于风格损失,Gatys 等人使用了层激活的格拉姆矩阵(Gram matrix),
# 即某一层特征图的内积。这个内积可以被理解成表示该层特征之间相互关系的映射。这些特征
# 相互关系抓住了在特定空间尺度下模式的统计规律,从经验上来看,它对应于这个尺度上找到
# 的纹理的外观。
# 
# 因此,风格损失的目的是在风格参考图像与生成图像之间,在不同的层激活内保存相似的
# 内部相互关系。反过来,这保证了在风格参考图像与生成图像之间,不同空间尺度找到的纹理看起来都很相似。
# 简而言之,你可以使用预训练的卷积神经网络来定义一个具有以下特点的损失。
#  在目标内容图像和生成图像之间保持相似的较高层激活,从而能够保留内容。卷积神经网络应该能够“看到”目标图像和生成图像包含相同的内容。
#  在较低层和较高层的激活中保持类似的相互关系(correlation),从而能够保留风格。特征相互关系捕捉到的是纹理(texture),生成图像和风格参考图像在不同的空间尺度上应该具有相同的纹理。

# ## 用 Keras 实现神经风格迁移
# 神经风格迁移可以用任何预训练卷积神经网络来实现。我们这里将使用 Gatys 等人所使用
# 的 VGG19 网络。VGG19 是第 5 章介绍的 VGG16 网络的简单变体,增加了三个卷积层。
# 神经风格迁移的一般过程如下。
# (1) 创建一个网络,它能够同时计算风格参考图像、目标图像和生成图像的 VGG19 层激活。
# (2) 使用这三张图像上计算的层激活来定义之前所述的损失函数,为了实现风格迁移,需要将这个损失函数最小化。
# (3) 设置梯度下降过程来将这个损失函数最小化。
# 我们首先来定义风格参考图像和目标图像的路径。为了确保处理后的图像具有相似的尺寸(如果图像尺寸差异很大,会使得风格迁移变得更加困难),
# 稍后需要将所有图像的高度调整为400 像素。

# In[2]:


from keras.preprocessing.image import load_img, img_to_array

# 想要变换的图像的路径
target_image_path = '/home/ubuntu/data/portrait.png'
# 风格图像的路径
style_reference_image_path = '/home/ubuntu/data/popova.jpg'

# 生成图像的尺寸
width, height = load_img(target_image_path).size
img_height = 400
img_width = int(width * img_height / height)


# 我们需要一些辅助函数,用于对进出 VGG19 卷积神经网络的图像进行加载、预处理和后处理。

# In[3]:


import numpy as np
from keras.applications import vgg19

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(img_height, img_width))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img

def deprocess_image(x):
    # vgg19.preprocess_input 的作用是减去 ImageNet 的平均像素值,使其中心为 0。这里相当于 vgg19.preprocess_input 的逆操作
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 将图像由 BGR 格式转换为 RGB 格式。这也是vgg19.preprocess_input 逆操作的一部分
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x


# 下面构建 VGG19 网络。它接收三张图像的批量作为输入,三张图像分别是风格参考图像、
# 目标图像和一个用于保存生成图像的占位符。占位符是一个符号张量,它的值由外部 Numpy 张
# 量提供。风格参考图像和目标图像都是不变的,因此使用 K.constant 来定义,但生成图像的
# 占位符所包含的值会随着时间而改变。

# In[4]:


from keras import backend as K

target_image = K.constant(preprocess_image(target_image_path))
style_reference_image = K.constant(preprocess_image(style_reference_image_path))

# 这个占位符用于保存生成图像
combination_image = K.placeholder((1, img_height, img_width, 3))

# 将三张图像合并为一个批量
input_tensor = K.concatenate([target_image,
                              style_reference_image,
                              combination_image], axis=0)

# 利用三张图像组成的批量作为输入来构建 VGG19 网络。加载模型将使用预训练的 ImageNet 权重
model = vgg19.VGG19(input_tensor=input_tensor,
                    weights='imagenet',
                    include_top=False)
print('Model loaded.')


# 我们来定义内容损失,它要保证目标图像和生成图像在 VGG19 卷积神经网络的顶层具有相似的结果。

# In[5]:


def content_loss(base, combination):
    return K.sum(K.square(combination - base))


# 接下来是风格损失。它使用一个辅助函数来计算输入矩阵的格拉姆矩阵,即原始特征矩阵中相互关系的映射。

# In[6]:


def gram_matrix(x):
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram


def style_loss(style, combination):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = img_height * img_width
    return K.sum(K.square(S - C)) / (4. * (channels ** 2) * (size ** 2))


# 除了这两个损失分量,我们还要添加第三个——总变差损失(total variation loss),它对生成的组合图像的像素进行操作。
# 它促使生成图像具有空间连续性,从而避免结果过度像素化。你可以将其理解为正则化损失。

# In[7]:


def total_variation_loss(x):
    a = K.square(
        x[:, :img_height - 1, :img_width - 1, :] - x[:, 1:, :img_width - 1, :])
    b = K.square(
        x[:, :img_height - 1, :img_width - 1, :] - x[:, :img_height - 1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))


# 我们需要最小化的损失是这三项损失的加权平均。为了计算内容损失,我们只使用一个靠
# 顶部的层,即 block5_conv2 层;而对于风格损失,我们需要使用一系列层,既包括顶层也包
# 括底层。最后还需要添加总变差损失。
# 根据所使用的风格参考图像和内容图像,很可能还需要调节 content_weight 系数(内容
# 损失对总损失的贡献比例)。更大的 content_weight 表示目标内容更容易在生成图像中被识
# 别出来。

# In[8]:


# 将层的名称映射为激活张量的字典
outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])
# 用于内容损失的层
content_layer = 'block5_conv2'
# 用于风格损失的层
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']
# 损失分量的加权平均所使用的权重
total_variation_weight = 1e-4
style_weight = 1.
content_weight = 0.025

# 添加内容损失
# 在定义损失时将所有分量添加到这个标量变量中
loss = K.variable(0.)
layer_features = outputs_dict[content_layer]
target_image_features = layer_features[0, :, :, :]
combination_features = layer_features[2, :, :, :]
loss += content_weight * content_loss(target_image_features,
                                      combination_features)

# 添加每个目标层的风格损失分量
for layer_name in style_layers:
    layer_features = outputs_dict[layer_name]
    style_reference_features = layer_features[1, :, :, :]
    combination_features = layer_features[2, :, :, :]
    sl = style_loss(style_reference_features, combination_features)
    loss += (style_weight / len(style_layers)) * sl
# 添加总变差损失
loss += total_variation_weight * total_variation_loss(combination_image)


# 最后需要设置梯度下降过程。在 Gatys 等人最初的论文中,使用 L-BFGS 算法进行最优化,
# 所以我们这里也将使用这种方法。这是本例与 8.2 节 DeepDream 例子的主要区别。L-BFGS 算
# 法内置于 SciPy 中,但 SciPy 实现有两个小小的限制。
#  它需要将损失函数值和梯度值作为两个单独的函数传入。
#  它只能应用于展平的向量,而我们的数据是三维图像数组。
# 分别计算损失函数值和梯度值是很低效的,因为这么做会导致二者之间大量的冗余计算。
# 这一过程需要的时间几乎是联合计算二者所需时间的 2 倍。为了避免这种情况,我们将创建一
# 个名为 Evaluator 的 Python 类,它可以同时计算损失值和梯度值,在第一次调用时会返回损
# 失值,同时缓存梯度值用于下一次调用。


# 获取损失相对于生成图像的梯度
grads = K.gradients(loss, combination_image)[0]

# 用于获取当前损失值和当前梯度值的函数
fetch_loss_and_grads = K.function([combination_image], [loss, grads])


# 这 个 类 将 fetch_loss_and_grads 包装起来,让你可以利用两个单独的方法调用来获取损失和梯度,这是我们要使用的 SciPy 优化器所要求的
class Evaluator(object):

    def __init__(self):
        self.loss_value = None
        self.grads_values = None

    def loss(self, x):
        assert self.loss_value is None
        x = x.reshape((1, img_height, img_width, 3))
        outs = fetch_loss_and_grads([x])
        loss_value = outs[0]
        grad_values = outs[1].flatten().astype('float64')
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values

evaluator = Evaluator()


# 最后,可以使用 SciPy 的 L-BFGS 算法来运行梯度上升过程,在算法每一次迭代时都保存当前的生成图像(这里一次迭代表示 20 个梯度上升步骤)。


from scipy.optimize import fmin_l_bfgs_b
from scipy.misc import imsave
import time

result_prefix = 'style_transfer_result'
iterations = 20

# 这是初始状态:目标图像
x = preprocess_image(target_image_path)

# 将图像展平,因为 scipy.optimize.fmin_l_bfgs_b 只能处理展平的向量
x = x.flatten()
for i in range(iterations):
    print('Start of iteration', i)
    start_time = time.time()
    # 对生成图像的像素运行L-BFGS 最优化,以将神经风格损失最小化。注意,必须将计算损失的函数和计算梯度的函数作为两个单独的参数传入
    x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x,
                                     fprime=evaluator.grads, maxfun=20)
    print('Current loss value:', min_val)
    # 保存当前的生成图像
    img = x.copy().reshape((img_height, img_width, 3))
    img = deprocess_image(img)
    fname = result_prefix + '_at_iteration_%d.png' % i
    imsave(fname, img)
    end_time = time.time()
    print('Image saved as', fname)
    print('Iteration %d completed in %ds' % (i, end_time - start_time))


# Here's what we get:

# In[14]:


from matplotlib import pyplot as plt

# Content image
plt.imshow(load_img(target_image_path, target_size=(img_height, img_width)))
plt.figure()

# Style image
plt.imshow(load_img(style_reference_image_path, target_size=(img_height, img_width)))
plt.figure()

# Generate image
plt.imshow(img)
plt.show()


# 请记住,这种技术所实现的仅仅是一种形式的改变图像纹理,
# 或者叫纹理迁移。如果风格参考图像具有明显的纹理结构且高度自相似,并且内容目标不需要
# 高层次细节就能够被识别,那么这种方法的效果最好。它通常无法实现比较抽象的迁移,比如
# 将一幅肖像的风格迁移到另一幅中。这种算法更接近于经典的信号处理,而不是更接近于人工智能,因此不要指望它能实现魔法般的效果。
#
# 此外还请注意,这个风格迁移算法的运行速度很慢。但这种方法实现的变换足够简单,只
# 要有适量的训练数据,一个小型的快速前馈卷积神经网络就可以学会这种变换。因此,实现快
# 速风格迁移的方法是,首先利用这里介绍的方法,花费大量的计算时间对一张固定的风格参考
# 图像生成许多输入 - 输出训练样例,然后训练一个简单的卷积神经网络来学习这个特定风格的
# 变换。一旦完成之后,对一张图像进行风格迁移是非常快的,只是这个小型卷积神经网络的一
# 次前向传递而已。
#
# 小结
#  风格迁移是指创建一张新图像,保留目标图像的内容的同时还抓住了参考图像的风格。
#  内容可以被卷积神经网络更靠顶部的层激活所捕捉到。
#  风格可以被卷积神经网络不同层激活的内部相互关系所捕捉到。
#  因此,深度学习可以将风格迁移表述为一个最优化过程,并用到了一个用预训练卷积神经网络所定义的损失。
#  从这个基本想法出发,可以有许多变体和改进
