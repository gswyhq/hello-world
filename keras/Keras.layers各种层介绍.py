#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import keras
from keras.layers import *

# 1.1、Dense层(全连接层）
keras.layers.core.Dense(units,
                        activation=None,
                        use_bias=True,kernel_initializer='glorot_uniform',bias_initializer='zeros',kernel_regularizer=None,
                        bias_regularizer=None,activity_regularizer=None,kernel_constraint=None,bias_constraint=None)

# 参数：
# units：大于0的整数，代表该层的输出维度。
# use_bias：布尔值，是否使用偏置项
# kernel_initializer：权值初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。
# bias_initializer：偏置向量初始化方法，为预定义初始化方法名的字符串，或用于初始化偏置向量的初始化器。
# regularizer：正则项，kernel为权重的、bias为偏执的，activity为输出的
# constraints：约束项，kernel为权重的，bias为偏执的。
# activation：激活函数，为预定义的激活函数名（参考激活函数），或逐元素（element-wise）的Theano函数。如果不指定该参数，将不会使用任何激活函数（即使用线性激活函数：a(x)=x）
# input_dim：该层输入的维度
# 本层实现的运算为
# output=activation(dot(input,kernel)+bias)


# 1.2、Activation层
keras.layers.core.Activation(activation)

# 激活层对一个层的输出施加激活函数
# 参数：
# activation：将要使用的激活函数，为预定义激活函数名或一个Tensorflow/Theano的函数。参考激活函数
# 输入shape：任意，当使用激活层作为第一层时，要指定input_shape
# 输出shape：与输入shape相同

# 1.3、dropout层
keras.layers.core.Dropout(rate, noise_shape=None, seed=None)

# 为输入数据施加Dropout。Dropout将在训练过程中每次更新参数时按一定概率（rate）随机断开输入神经元，Dropout层用于防止过拟合。
# 参数
# rate：0~1的浮点数，控制需要断开的神经元的比例
# noise_shape：整数张量，为将要应用在输入上的二值Dropout mask的shape，例如你的输入为(batch_size, timesteps, features)，并且你希望在各个时间步上的Dropout mask都相同，则可传入noise_shape=(batch_size, 1, features)。
# seed：整数，使用的随机数种子

# 1.4、Flatten层
keras.layers.core.Flatten()

# Flatten层用来将输入“压平”，即把多维的输入一维化，常用在从卷积层到全连接层的过渡。Flatten不影响batch的大小。
# batch, a, b, c -> batch, a*b*c
x = np.random.random(size=(2,3,4))
# x
# Out[36]:
# array([[[0.4414674 , 0.47956157, 0.95838914, 0.79389298],
#         [0.93795243, 0.05868621, 0.75216682, 0.21120361],
#         [0.72700799, 0.76903055, 0.43342402, 0.67986415]],
#        [[0.90176034, 0.35007762, 0.5242487 , 0.88207313],
#         [0.30160922, 0.53073638, 0.07346707, 0.51390339],
#         [0.16652861, 0.50185009, 0.53312969, 0.43941677]]])
Flatten()(x)
# Out[37]:
# <tf.Tensor: shape=(2, 12), dtype=float32, numpy=
# array([[0.4414674 , 0.47956157, 0.95838916, 0.793893  , 0.93795246,
#         0.05868621, 0.7521668 , 0.21120362, 0.727008  , 0.7690306 ,
#         0.43342403, 0.67986417],
#        [0.90176034, 0.35007763, 0.5242487 , 0.8820731 , 0.30160922,
#         0.5307364 , 0.07346708, 0.5139034 , 0.16652861, 0.50185007,
#         0.5331297 , 0.43941677]], dtype=float32)>


# 1.5、Reshape层
keras.layers.core.Reshape(target_shape)

# Reshape层用来将输入shape转换为特定的shape
#
# 参数
# target_shape：目标shape，为整数的tuple，不包含样本数目的维度（batch大小）
# 输入shape：任意，但输入的shape必须固定。当使用该层为模型首层时，需要指定input_shape参数
# 输出shape：(batch_size,)+target_shape

# 1.6、Permute层
keras.layers.core.Permute(dims)

# Permute层将输入的维度按照给定模式进行重排，例如，当需要将RNN和CNN网络连接时，可能会用到该层。所谓的重排也就是交换两行

# 参数
# dims：整数tuple，指定重排的模式，不包含样本数batch的维度(下标是0)。重拍模式的下标从1开始。例如（2，1）代表将输入的第二个维度重排到输出的第一个维度，而将输入的第一个维度重排到第二个维度

# 输入shape：任意，当使用激活层作为第一层时，要指定input_shape
# 输出shape：与输入相同，但是其维度按照指定的模式重新排列

# Reshape、Permute的异同：
# 相同点：二者都可以输出相同的维度
# 异同点：Reshape是重新排列数据，Permute则是对应维度数据交换；
x = np.random.random(size=(1, 2, 3))
# x
# Out[39]:
# array([[[0.19943505, 0.26292927, 0.36845205],
#         [0.9312268 , 0.9974761 , 0.94398579]]])
Reshape(target_shape=(3, 2))(x)
# Out[40]:
# <tf.Tensor: shape=(1, 3, 2), dtype=float32, numpy=
# array([[[0.19943506, 0.26292926],
#         [0.36845204, 0.9312268 ],
#         [0.9974761 , 0.94398576]]], dtype=float32)>
Permute(dims=(2, 1))(x)
# Out[41]:
# <tf.Tensor: shape=(1, 3, 2), dtype=float32, numpy=
# array([[[0.19943506, 0.9312268 ],
#         [0.26292926, 0.9974761 ],
#         [0.36845204, 0.94398576]]], dtype=float32)>

# 1.7、RepeatVector层
keras.layers.core.RepeatVector(n)

# RepeatVector层将输入重复n次
# 参数
# n：整数，重复的次数
# 输入shape：形如（nb_samples, features）的2D张量
# 输出shape：形如（nb_samples, n, features）的3D张量
# 例子
# x = np.random.random(size=(1, 2))
# x
# Out[46]: array([[0.59371616, 0.24741683]])
# RepeatVector(3)(x)
# Out[47]:
# <tf.Tensor: shape=(1, 3, 2), dtype=float32, numpy=
# array([[[0.59371614, 0.24741682],
#         [0.59371614, 0.24741682],
#         [0.59371614, 0.24741682]]], dtype=float32)>

# 1.8、Lambda层
keras.layers.core.Lambda(function, output_shape=None, mask=None, arguments=None)

# 本函数用以对上一层的输出施以任何Theano/TensorFlow表达式
# 参数
# function：要实现的函数，该函数仅接受一个变量，即上一层的输出
# output_shape：函数应该返回的值的shape，可以是一个tuple，也可以是一个根据输入
# shape计算输出shape的函数
# mask: 掩膜
# arguments：可选，字典，用来记录向函数中传递的其他关键字参数
# 输入shape：任意，当使用该层作为第一层时，要指定input_shape
# 输出shape：由output_shape参数指定的输出shape，当使用tensorflow时可自动推断

# x = np.random.random(size=(1, 2))
# x
# Out[46]: array([[0.59371616, 0.24741683]])
# Lambda(lambda x: tf.einsum('ij,ij->ij', x[0], x[1]))([x, x])
# Out[48]: <tf.Tensor: shape=(1, 2), dtype=float32, numpy=array([[0.35249886, 0.06121508]], dtype=float32)>

# 1.9、ActivityRegularizer层
keras.layers.core.ActivityRegularization(l1=0.0, l2=0.0)

# 经过本层的数据不会有任何变化，但会基于其激活值更新损失函数值
# 参数

# l1：1范数正则因子（正浮点数）
# l2：2范数正则因子（正浮点数）
# 输入shape：任意，当使用该层作为第一层时，要指定input_shape
# 输出shape：与输入shape相同

# 2.0、Masking层
keras.layers.core.Masking(mask_value=0.0)

# 2.1、Conv1D层
keras.layers.convolutional.Conv1D(filters, kernel_size, strides=1,
                                  padding='valid', dilation_rate=1, activation=None, use_bias=True,
                                  kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None,
                                  bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)

# 一维卷积层（即时域卷积），用以在一维输入信号上进行邻域滤波。当使用该层作为首层时，需要提供关键字参数input_shape。例如(10,128)代表一个长为10的序列，序列中每个信号为128向量。而(None, 128)代表变长的128维向量序列。
# 该层生成将输入信号与卷积核按照单一的空域（或时域）方向进行卷积。如果use_bias=True，则还会加上一个偏置项，若activation不为None，则输出为经过激活函数的输出。

# 参数
# filters：卷积核的数目（即输出的维度）
# kernel_size：整数或由单个整数构成的list/tuple，卷积核的空域或时域窗长度
# strides：整数或由单个整数构成的list/tuple，为卷积的步长。任何不为1的strides均与任何不为1的dilation_rate均不兼容
# padding：补0策略，为“valid”, “same” 或“causal”，“causal”将产生因果（膨胀的）卷积，即output[t]不依赖于input[t+1：]。当对不能违反时间顺序的时序信号建模时有用。
# “valid”代表只进行有效的卷积，即对边界数据不处理。“same”代表保留边界处的卷积结果，通常会导致输出shape与输入shape相同。
# activation：激活函数，为预定义的激活函数名（参考激活函数），或逐元素（element-wise）的Theano函数。如果不指定该参数，将不会使用任何激活函数（即使用线性激活函数：a(x)=x）
# dilation_rate：整数或由单个整数构成的list/tuple，指定dilated convolution中的膨胀比例。任何不为1的dilation_rate均与任何不为1的strides均不兼容。
# use_bias:布尔值，是否使用偏置项
# kernel_initializer：权值初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。参考initializers
# bias_initializer：权值初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。参考initializers
# kernel_regularizer：施加在权重上的正则项，为Regularizer对象
# bias_regularizer：施加在偏置向量上的正则项，为Regularizer对象
# activity_regularizer：施加在输出上的正则项，为Regularizer对象
# kernel_constraints：施加在权重上的约束项，为Constraints对象
# bias_constraints：施加在偏置上的约束项，为Constraints对象
# 输入shape：形如（samples，steps，input_dim）的3D张量
# 输出shape：形如（samples，new_steps，nb_filter）的3D张量，因为有向量填充的原因，steps的值会改变

# 【Tips】可以将Convolution1D看作Convolution2D的快捷版，对例子中（10，32）的信号进行1D卷积相当于对其进行卷积核为（filter_length, 32）的2D卷积。

# Conv1D的计算过程：
x = np.random.random(size=(1, 4, 3))
# x
# Out[57]:
# array([[[0.92807342, 0.03695125, 0.71480237],
#         [0.37405272, 0.69259756, 0.56800219],
#         [0.99615874, 0.62926799, 0.01772351],
#         [0.20668637, 0.19838596, 0.48282641]]])
conv = Conv1D(4, 2)  # 定义4个卷积核(即最后输出的结果中最后一个维度为4)，卷积核域窗长度为2
conv(x)
# Out[56]:
# <tf.Tensor: shape=(1, 3, 4), dtype=float32, numpy=
# array([[[ 0.11789236, -0.3760438 ,  0.7823301 ,  0.84105694],
#         [ 0.97095454,  0.25242725,  0.54123855,  0.39013147],
#         [ 0.03222343,  0.23442352,  0.39691785, -0.09012049]]],
#       dtype=float32)>
# conv.weights
# Out[58]:
# [<tf.Variable 'conv1d_14/kernel:0' shape=(2, 3, 4) dtype=float32, numpy=
#  array([[[-0.39086345, -0.34305924,  0.2140826 ,  0.31867164],
#          [ 0.599702  ,  0.6061559 , -0.02700508, -0.44864124],
#          [ 0.6378293 , -0.42888772,  0.47828412,  0.6366242 ]],
#
#         [[ 0.47737193,  0.26218683,  0.22007364, -0.16908896],
#          [-0.21480697, -0.09981108, -0.02639169,  0.6305995 ],
#          [-0.04793435,  0.34783208,  0.3146549 , -0.469505  ]]],
#        dtype=float32)>,
#  <tf.Variable 'conv1d_14/bias:0' shape=(4,) dtype=float32, numpy=array([0., 0., 0., 0.], dtype=float32)>]
# output, 当strides步长为1，padding=“valid”时, 输出的维度shape = (batch, x.shape[-2]-kernel_size+1, filters)
# weights的shape=(2,3,4), shape = (kernel_size, x.shape[-1], 卷积核个数)
# 输出结果计算过程：
# 第一个卷积核：conv.weights[0][:,:,0]
# <tf.Tensor: shape=(2, 3), dtype=float32, numpy=
# array([[-0.39086345,  0.599702  ,  0.6378293 ],
#        [ 0.47737193, -0.21480697, -0.04793435]], dtype=float32)>
# 第一个卷积核对应的值计算(注意：这里bias=0，故结果是对应数值按位相乘再求和即可)：
# 因卷积核域窗长度为2，故依次取2列值与卷积核权重进行按位相乘再求和
# tf.einsum('ij,ij->', x[0,:2,:], conv.weights[0][:,:,0])
# Out[106]: <tf.Tensor: shape=(), dtype=float32, numpy=0.117892385>
# tf.einsum('ij,ij->', x[0,1:3,:], conv.weights[0][:,:,0])
# Out[107]: <tf.Tensor: shape=(), dtype=float32, numpy=0.9709545>
# tf.einsum('ij,ij->', x[0,2:4,:], conv.weights[0][:,:,0])
# Out[108]: <tf.Tensor: shape=(), dtype=float32, numpy=0.032223433>

# 同理，第二个卷积核对应结果计算：
# tf.einsum('ij,ij->', x[0,:2,:], conv.weights[0][:,:,1])
# Out[109]: <tf.Tensor: shape=(), dtype=float32, numpy=-0.37604374>
# tf.einsum('ij,ij->', x[0,1:3,:], conv.weights[0][:,:,1])
# Out[110]: <tf.Tensor: shape=(), dtype=float32, numpy=0.25242728>
# tf.einsum('ij,ij->', x[0,2:4,:], conv.weights[0][:,:,1])
# Out[111]: <tf.Tensor: shape=(), dtype=float32, numpy=0.23442352>

# 若输入参数增加一个维度时，对应输入结果，保留output.shape[1],如：
x = np.random.random(size=(1, 4, 3, 2))
conv = Conv1D(4, 3)  # 定义4个卷积核(即最后输出的结果中最后一个维度为4)，卷积核域窗长度为3
conv(x)  # output.shape = (x.shape[0], x.shape[1], x.shape[2]-kernel_size+1, filters)
# Out[117]:
# <tf.Tensor: shape=(1, 4, 1, 4), dtype=float32, numpy=
# array([[[[ 0.0373335 , -0.421912  ,  0.38562953, -0.01787274]],
#         [[ 0.4232917 , -0.67487305,  0.56536055, -0.31096753]],
#         [[ 0.2459468 , -0.778931  ,  0.5654522 , -0.37960288]],
#         [[ 0.5973935 , -0.4109627 ,  0.01628786, -0.19279176]]]],
#       dtype=float32)>
# conv.weights
# Out[118]:
# [<tf.Variable 'conv1d_17/kernel:0' shape=(3, 2, 4) dtype=float32, numpy=
#  array([[[ 0.2002083 , -0.5005008 ,  0.43203735, -0.18884125],
#          [ 0.50688386, -0.5364116 , -0.47853857,  0.01282918]],
#
#         [[-0.24771294,  0.562464  ,  0.22159928,  0.3925528 ],
#          [-0.34466538, -0.3196959 , -0.07459968,  0.06492996]],
#
#         [[ 0.3839057 ,  0.24000639,  0.3389361 , -0.37646484],
#          [ 0.18893391, -0.5748589 ,  0.1735866 , -0.28434834]]],
#        dtype=float32)>,
#  <tf.Variable 'conv1d_17/bias:0' shape=(4,) dtype=float32, numpy=array([0., 0., 0., 0.], dtype=float32)>]
# tf.einsum("ij,ij->", x[0,0,:,:], conv.weights[0][:,:,0])
# Out[119]: <tf.Tensor: shape=(), dtype=float32, numpy=0.03733349>

# 2.2、Conv2D层
keras.layers.convolutional.Conv2D(filters, kernel_size, strides=(1, 1),
                                  padding='valid', data_format=None, dilation_rate=(1, 1), activation=None,
                                  use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros',
                                  kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)

# 二维卷积层，即对图像的空域卷积。该层对二维输入进行滑动窗卷积，当使用该层作为第一层时，应提供input_shape参数。例如input_shape = (128,128,3)代表128*128的彩色RGB图像（data_format=‘channels_last’）
# 参数
# filters：卷积核的数目（即输出的维度）
# kernel_size：单个整数或由两个整数构成的list/tuple，卷积核的宽度和长度。如为单个整数，则表示在各个空间维度的相同长度。
# strides：单个整数或由两个整数构成的list/tuple，为卷积的步长。如为单个整数，则表示在各个空间维度的相同步长。任何不为1的strides均与任何不为1的dilation_rate均不兼容
# padding：补0策略，为“valid”, “same” 。“valid”代表只进行有效的卷积，即对边界数据不处理。“same”代表保留边界处的卷积结果，通常会导致输出shape与输入shape相同。
# activation：激活函数，为预定义的激活函数名（参考激活函数），或逐元素（element-wise）的Theano函数。如果不指定该参数，将不会使用任何激活函数（即使用线性激活函数：a(x)=x）
# dilation_rate：单个整数或由两个整数构成的list/tuple，指定dilated convolution中的膨胀比例。任何不为1的dilation_rate均与任何不为1的strides均不兼容。
# data_format：字符串，“channels_first”或“channels_last”之一，代表图像的通道维的位置。该参数是Keras 1.x中的image_dim_ordering，“channels_last”对应原本的“tf”，“channels_first”对应原本的“th”。以128x128的RGB图像为例，“channels_first”应将数据组织为（3,128,128），而“channels_last”应将数据组织为（128,128,3）。该参数的默认值是~/.keras/keras.json中设置的值，若从未设置过，则为“channels_last”。
# use_bias:布尔值，是否使用偏置项
# kernel_initializer：权值初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。参考initializers
# bias_initializer：权值初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。参考initializers
# kernel_regularizer：施加在权重上的正则项，为Regularizer对象
# bias_regularizer：施加在偏置向量上的正则项，为Regularizer对象
# activity_regularizer：施加在输出上的正则项，为Regularizer对象
# kernel_constraints：施加在权重上的约束项，为Constraints对象
# bias_constraints：施加在偏置上的约束项，为Constraints对象
# 输入shape：
# ‘channels_first’模式下，输入形如（samples,channels，rows，cols）的4D张量。
# ‘channels_last’模式下，输入形如（samples，rows，cols，channels）的4D张量。

# 注意这里的输入shape指的是函数内部实现的输入shape，而非函数接口应指定的input_shape，请参考下面提供的例子。
#
# 输出shape：
# ‘channels_first’模式下，为形如（samples，nb_filter, new_rows, new_cols）的4D张量。
# ‘channels_last’模式下，为形如（samples，new_rows, new_cols，nb_filter）的4D张。量
#
# 输出的行列数可能会因为填充方法而改变。

# 2.3、SeparableConv2D层
keras.layers.convolutional.SeparableConv2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None,
                                           depth_multiplier=1, activation=None, use_bias=True, depthwise_initializer='glorot_uniform',
                                           pointwise_initializer='glorot_uniform', bias_initializer='zeros', depthwise_regularizer=None,
                                           pointwise_regularizer=None, bias_regularizer=None, activity_regularizer=None, depthwise_constraint=None,
                                           pointwise_constraint=None, bias_constraint=None)

# 该层是在深度方向上的可分离卷积。
# 可分离卷积首先按深度方向进行卷积（对每个输入通道分别卷积），然后逐点进行卷积，将上一步的卷积结果混合到输出通道中。参数depth_multiplier控制了在depthwise卷积（第一步）的过程中，每个输入通道信号产生多少个输出通道。
# 直观来说，可分离卷积可以看做讲一个卷积核分解为两个小的卷积核，或看作Inception模块的一种极端情况。
# 当使用该层作为第一层时，应提供input_shape参数。例如input_shape = (3,128,128)代表128*128的彩色RGB图像。
# 参数
# filters：卷积核的数目（即输出的维度）
# kernel_size：单个整数或由两个个整数构成的list/tuple，卷积核的宽度和长度。如为单个整数，则表示在各个空间维度的相同长度。
# strides：单个整数或由两个整数构成的list/tuple，为卷积的步长。如为单个整数，则表示在各个空间维度的相同步长。任何不为1的strides均与任何不为1的dilation_rate均不兼容
# padding：补0策略，为“valid”, “same”
# 。“valid”代表只进行有效的卷积，即对边界数据不处理。“same”代表保留边界处的卷积结果，通常会导致输出shape与输入shape相同。
# activation：激活函数，为预定义的激活函数名（参考激活函数），或逐元素（element-wise）的Theano函数。如果不指定该参数，将不会使用任何激活函数（即使用线性激活函数：a(x)=x）
# dilation_rate：单个整数或由两个整数构成的list/tuple，指定dilated
# convolution中的膨胀比例。任何不为1的dilation_rate均与任何不为1的strides均不兼容。
# data_format：字符串，“channels_first”或“channels_last”之一，代表图像的通道维的位置。该参数是Keras1.x中的image_dim_ordering，“channels_last”对应原本的“tf”，“channels_first”对应原本的“th”。以128x128的RGB图像为例，“channels_first”应将数据组织为（3,128,128），而“channels_last”应将数据组织为（128,128,3）。该参数的默认值是~/.keras/keras.json中设置的值，若从未设置过，则为“channels_last”。
# use_bias:布尔值，是否使用偏置项 depth_multiplier：在按深度卷积的步骤中，每个输入通道使用多少个输出通道
# kernel_initializer：权值初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。参考initializers
# bias_initializer：权值初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。参考initializers
# depthwise_regularizer：施加在按深度卷积的权重上的正则项，为Regularizer对象
# pointwise_regularizer：施加在按点卷积的权重上的正则项，为Regularizer对象
# kernel_regularizer：施加在权重上的正则项，为Regularizer对象
# bias_regularizer：施加在偏置向量上的正则项，为Regularizer对象
# activity_regularizer：施加在输出上的正则项，为Regularizer对象
# kernel_constraints：施加在权重上的约束项，为Constraints对象
# bias_constraints：施加在偏置上的约束项，为Constraints对象
# depthwise_constraint：施加在按深度卷积权重上的约束项，为Constraints对象
# pointwise_constraint施加在按点卷积权重的约束项，为Constraints对象
# 输入shape
# ‘channels_first’模式下，输入形如（samples,channels，rows，cols）的4D张量。
# ‘channels_last’模式下，输入形如（samples，rows，cols，channels）的4D张量。

# 注意这里的输入shape指的是函数内部实现的输入shape，而非函数接口应指定的input_shape，请参考下面提供的例子。

# 输出shape
# ‘channels_first’模式下，为形如（samples，nb_filter, new_rows, new_cols）的4D张量。
# ‘channels_last’模式下，为形如（samples，new_rows, new_cols，nb_filter）的4D张量。

# 输出的行列数可能会因为填充方法而改变

# 2.4、Conv2DTranspose层
keras.layers.convolutional.Conv2DTranspose(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None,
                                           activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros',
                                           kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)

# 该层是转置的卷积操作（反卷积）。需要反卷积的情况通常发生在用户想要对一个普通卷积的结果做反方向的变换。例如，将具有该卷积层输出shape的tensor转换为具有该卷积层输入shape的tensor。同时保留与卷积层兼容的连接模式。
# 当使用该层作为第一层时，应提供input_shape参数。例如input_shape = (3,128,128)代表128*128的彩色RGB图像。
# 参数
# filters：卷积核的数目（即输出的维度）
# kernel_size：单个整数或由两个个整数构成的list/tuple，卷积核的宽度和长度。如为单个整数，则表示在各个空间维度的相同长度。
# strides：单个整数或由两个整数构成的list/tuple，为卷积的步长。如为单个整数，则表示在各个空间维度的相同步长。任何不为1的strides均与任何不为1的dilation_rate均不兼容
# padding：补0策略，为“valid”, “same” 。“valid”代表只进行有效的卷积，即对边界数据不处理。“same”代表保留边界处的卷积结果，通常会导致输出shape与输入shape相同。
# activation：激活函数，为预定义的激活函数名（参考激活函数），或逐元素（element-wise）的Theano函数。如果不指定该参数，将不会使用任何激活函数（即使用线性激活函数：a(x)=x）
# dilation_rate：单个整数或由两个个整数构成的list/tuple，指定dilated convolution中的膨胀比例。任何不为1的dilation_rate均与任何不为1的strides均不兼容。
# data_format：字符串，“channels_first”或“channels_last”之一，代表图像的通道维的位置。该参数是Keras 1.x中的image_dim_ordering，“channels_last”对应原本的“tf”，“channels_first”对应原本的“th”。以128x128的RGB图像为例，“channels_first”应将数据组织为（3,128,128），而“channels_last”应将数据组织为（128,128,3）。该参数的默认值是~/.keras/keras.json中设置的值，若从未设置过，则为“channels_last”。
# use_bias:布尔值，是否使用偏置项
# kernel_initializer：权值初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。参考initializers
# bias_initializer：权值初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。参考initializers
# kernel_regularizer：施加在权重上的正则项，为Regularizer对象
# bias_regularizer：施加在偏置向量上的正则项，为Regularizer对象
# activity_regularizer：施加在输出上的正则项，为Regularizer对象
# kernel_constraints：施加在权重上的约束项，为Constraints对象
# bias_constraints：施加在偏置上的约束项，为Constraints对象
# 输入shape
# ‘channels_first’模式下，输入形如（samples,channels，rows，cols）的4D张量。
# ‘channels_last’模式下，输入形如（samples，rows，cols，channels）的4D张量。

# 注意这里的输入shape指的是函数内部实现的输入shape，而非函数接口应指定的input_shape，请参考下面提供的例子。

# 输出shape
# ‘channels_first’模式下，为形如（samples，nb_filter, new_rows, new_cols）的4D张量。
# ‘channels_last’模式下，为形如（samples，new_rows, new_cols，nb_filter）的4D张量。

# 输出的行列数可能会因为填充方法而改变

# 2.5、Conv3D层
keras.layers.convolutional.Conv3D(filters, kernel_size, strides=(1, 1, 1), padding='valid', data_format=None, dilation_rate=(1, 1, 1),
                                  activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros',
                                  kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)

# 三维卷积对三维的输入进行滑动窗卷积，当使用该层作为第一层时，应提供input_shape参数。例如input_shape = (3,10,128,128)代表对10帧128*128的彩色RGB图像进行卷积。数据的通道位置仍然有data_format参数指定。
# 参数
# filters：卷积核的数目（即输出的维度）
# kernel_size：单个整数或由3个整数构成的list/tuple，卷积核的宽度和长度。如为单个整数，则表示在各个空间维度的相同长度。
# strides：单个整数或由3个整数构成的list/tuple，为卷积的步长。如为单个整数，则表示在各个空间维度的相同步长。任何不为1的strides均与任何不为1的dilation_rate均不兼容
# padding：补0策略，为“valid”, “same” 。“valid”代表只进行有效的卷积，即对边界数据不处理。“same”代表保留边界处的卷积结果，通常会导致输出shape与输入shape相同。
# activation：激活函数，为预定义的激活函数名（参考激活函数），或逐元素（element-wise）的Theano函数。如果不指定该参数，将不会使用任何激活函数（即使用线性激活函数：a(x)=x）
# dilation_rate：单个整数或由3个个整数构成的list/tuple，指定dilated convolution中的膨胀比例。任何不为1的dilation_rate均与任何不为1的strides均不兼容。
# data_format：字符串，“channels_first”或“channels_last”之一，代表数据的通道维的位置。该参数是Keras 1.x中的image_dim_ordering，“channels_last”对应原本的“tf”，“channels_first”对应原本的“th”。以128x128x128的数据为例，“channels_first”应将数据组织为（3,128,128,128），而“channels_last”应将数据组织为（128,128,128,3）。该参数的默认值是~/.keras/keras.json中设置的值，若从未设置过，则为“channels_last”。
# use_bias:布尔值，是否使用偏置项
# kernel_initializer：权值初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。参考initializers
# bias_initializer：权值初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。参考initializers
# kernel_regularizer：施加在权重上的正则项，为Regularizer对象
# bias_regularizer：施加在偏置向量上的正则项，为Regularizer对象
# activity_regularizer：施加在输出上的正则项，为Regularizer对象
# kernel_constraints：施加在权重上的约束项，为Constraints对象
# bias_constraints：施加在偏置上的约束项，为Constraints对象
# 输入shape
# ‘channels_first’模式下，输入应为形如（samples，channels，input_dim1，input_dim2, input_dim3）的5D张量
# ‘channels_last’模式下，输入应为形如（samples，input_dim1，input_dim2, input_dim3，channels）的5D张量

# 这里的输入shape指的是函数内部实现的输入shape，而非函数接口应指定的input_shape。

# 2.6、Cropping1D层
keras.layers.convolutional.Cropping1D(cropping=(1, 1))

# 在时间轴（axis1）上对1D输入（即时间序列）进行裁剪
# 参数
# cropping：长为2的tuple，指定在序列的首尾要裁剪掉多少个元素
# 输入shape：形如（samples，axis_to_crop，features）的3D张量
# 输出shape：形如（samples，cropped_axis，features）的3D张量。

# 首尾各裁减一个元素的例子：
# x = np.random.random(size=(1, 4, 2))
# x
# Out[127]:
# array([[[0.00419808, 0.10018762],
#         [0.91153567, 0.29094981],
#         [0.83732935, 0.11907126],
#         [0.62115734, 0.86223209]]])
Cropping1D(cropping=(1, 1))(x)  # 首尾各裁减一个元素：
# Out[128]:
# <tf.Tensor: shape=(1, 2, 2), dtype=float32, numpy=
# array([[[0.9115357 , 0.29094982],
#         [0.8373293 , 0.11907126]]], dtype=float32)>

# 2.7、Cropping2D层
keras.layers.convolutional.Cropping2D(cropping=((0, 0), (0, 0)), data_format=None)

# 对2D输入（图像）进行裁剪，将在空域维度，即宽和高的方向上裁剪
# 参数
# cropping：长为2的整数tuple，分别为宽和高方向上头部与尾部需要裁剪掉的元素数
# data_format：字符串，“channels_first”或“channels_last”之一，代表图像的通道维的位置。该参数是Keras 1.x中的image_dim_ordering，“channels_last”对应原本的“tf”，“channels_first”对应原本的“th”。以128x128的RGB图像为例，“channels_first”应将数据组织为（3,128,128），而“channels_last”应将数据组织为（128,128,3）。该参数的默认值是~/.keras/keras.json中设置的值，若从未设置过，则为“channels_last”。
# 输入shape：形如（samples，depth, first_axis_to_crop, second_axis_to_crop）
# 输出shape：形如(samples, depth, first_cropped_axis, second_cropped_axis)的4D张量。

# 2.8、Cropping3D层
keras.layers.convolutional.Cropping3D(cropping=((1, 1), (1, 1), (1, 1)), data_format=None)

# 对2D输入（图像）进行裁剪
# 参数
# cropping：长为3的整数tuple，分别为三个方向上头部与尾部需要裁剪掉的元素数
# data_format：字符串，“channels_first”或“channels_last”之一，代表数据的通道维的位置。该参数是Keras 1.x中的image_dim_ordering，“channels_last”对应原本的“tf”，“channels_first”对应原本的“th”。以128x128x128的数据为例，“channels_first”应将数据组织为（3,128,128,128），而“channels_last”应将数据组织为（128,128,128,3）。该参数的默认值是~/.keras/keras.json中设置的值，若从未设置过，则为“channels_last”。
# 输入shape：形如 (samples, depth, first_axis_to_crop, second_axis_to_crop, third_axis_to_crop)的5D张量。
# 输出shape：形如(samples, depth, first_cropped_axis, second_cropped_axis, third_cropped_axis)的5D张量。

# 2.9、UpSampling1D层
keras.layers.convolutional.UpSampling1D(size=2)

# 在时间轴上，将每个时间步重复length次
# 参数
# size：上采样因子
# 输入shape：形如（samples，steps，features）的3D张量
# 输出shape：形如（samples，upsampled_steps，features）的3D张量

x = np.random.random(size=(1, 2, 3))
# x
# Out[130]:
# array([[[0.26908225, 0.49528201, 0.92729853],
#         [0.38189244, 0.65989714, 0.09706107]]])
UpSampling1D(size=2)(x)  # 输出shape= x.shape[0], x.shape[1]*size, x.shape[2]
# Out[131]:
# <tf.Tensor: shape=(1, 4, 3), dtype=float32, numpy=
# array([[[0.26908225, 0.49528202, 0.92729855],
#         [0.26908225, 0.49528202, 0.92729855],
#         [0.38189244, 0.65989715, 0.09706107],
#         [0.38189244, 0.65989715, 0.09706107]]], dtype=float32)>

# 3.0、UpSampling2D层
keras.layers.convolutional.UpSampling2D(size=(2, 2), data_format=None)

# 将数据的行和列分别重复size[0]和size[1]次
# 参数
# size：整数tuple，分别为行和列上采样因子
# data_format：字符串，“channels_first”或“channels_last”之一，代表图像的通道维的位置。该参数是Keras 1.x中的image_dim_ordering，“channels_last”对应原本的“tf”，“channels_first”对应原本的“th”。
# 以128x128的RGB图像为例，“channels_first”应将数据组织为（3,128,128），而“channels_last”应将数据组织为（128,128,3）。该参数的默认值是~/.keras/keras.json中设置的值，若从未设置过，则为“channels_last”。
# 输入shape：
# ‘channels_first’模式下，为形如（samples，channels, rows，cols）的4D张量。
# ‘channels_last’模式下，为形如（samples，rows, cols，channels）的4D张量。
# 输出shape：
# ‘channels_first’模式下，为形如（samples，channels, upsampled_rows, upsampled_cols）的4D张量。
# ‘channels_last’模式下，为形如（samples，upsampled_rows, upsampled_cols，channels）的4D张量。

# 3.1、UpSampling3D层
keras.layers.convolutional.UpSampling3D(size=(2, 2, 2), data_format=None)

# 将数据的三个维度上分别重复size[0]、size[1]和ize[2]次
# 本层目前只能在使用Theano为后端时可用
# 参数
# size：长为3的整数tuple，代表在三个维度上的上采样因子
# data_format：字符串，“channels_first”或“channels_last”之一，代表数据的通道维的位置。该参数是Keras 1.x中的image_dim_ordering，“channels_last”对应原本的“tf”，“channels_first”对应原本的“th”。
# 以128x128x128的数据为例，“channels_first”应将数据组织为（3,128,128,128），而“channels_last”应将数据组织为（128,128,128,3）。
# 该参数的默认值是~/.keras/keras.json中设置的值，若从未设置过，则为“channels_last”。
# 输入shape：
# ‘channels_first’模式下，为形如（samples, channels, len_pool_dim1, len_pool_dim2, len_pool_dim3）的5D张量
# ‘channels_last’模式下，为形如（samples, len_pool_dim1, len_pool_dim2, len_pool_dim3，channels, ）的5D张量
#
# 输出shape：
# ‘channels_first’模式下，为形如（samples, channels, dim1, dim2, dim3）的5D张量
# ‘channels_last’模式下，为形如（samples, upsampled_dim1, upsampled_dim2, upsampled_dim3,channels,）的5D张量。

# 3.2、ZeroPadding1D层
keras.layers.convolutional.ZeroPadding1D(padding=1)

# 对1D输入的首尾端（如时域序列）填充0，以控制卷积以后向量的长度
# 参数
# padding：整数，表示在要填充的轴的起始和结束处填充0的数目，这里要填充的轴是轴1（第1维，第0维是样本数）
# 输入shape：形如（samples，axis_to_pad，features）的3D张量
# 输出shape：形如（samples，paded_axis，features）的3D张量

# x
# Out[132]:
# array([[[0.26908225, 0.49528201, 0.92729853],
#         [0.38189244, 0.65989714, 0.09706107]]])
# ZeroPadding1D(padding=1)(x)
# Out[133]:
# <tf.Tensor: shape=(1, 4, 3), dtype=float32, numpy=
# array([[[0.        , 0.        , 0.        ],
#         [0.26908225, 0.49528202, 0.92729855],
#         [0.38189244, 0.65989715, 0.09706107],
#         [0.        , 0.        , 0.        ]]], dtype=float32)>

# 3.3、ZeroPadding2D层
keras.layers.convolutional.ZeroPadding2D(padding=(1, 1), data_format=None)

# 对2D输入（如图片）的边界填充0，以控制卷积以后特征图的大小
# 参数
# padding：整数tuple，表示在要填充的轴的起始和结束处填充0的数目，这里要填充的轴是轴3和轴4（即在’th’模式下图像的行和列，在‘channels_last’模式下要填充的则是轴2，3）
# data_format：字符串，“channels_first”或“channels_last”之一，代表图像的通道维的位置。该参数是Keras 1.x中的image_dim_ordering，“channels_last”对应原本的“tf”，“channels_first”对应原本的“th”。以128x128的RGB图像为例，“channels_first”应将数据组织为（3,128,128），而“channels_last”应将数据组织为（128,128,3）。该参数的默认值是~/.keras/keras.json中设置的值，若从未设置过，则为“channels_last”。
# 输入shape：
# ‘channels_first’模式下，形如（samples，channels，first_axis_to_pad，second_axis_to_pad）的4D张量。
# ‘channels_last’模式下，形如（samples，first_axis_to_pad，second_axis_to_pad, channels）的4D张量。
# 输出shape：
# ‘channels_first’模式下，形如（samples，channels，first_paded_axis，second_paded_axis）的4D张量
# ‘channels_last’模式下，形如（samples，first_paded_axis，second_paded_axis, channels）的4D张量

# 3.4、ZeroPadding3D层
keras.layers.convolutional.ZeroPadding3D(padding=(1, 1, 1), data_format=None)

# 将数据的三个维度上填充0
# 本层目前只能在使用Theano为后端时可用
# 参数
# padding：整数tuple，表示在要填充的轴的起始和结束处填充0的数目，这里要填充的轴是轴3，轴4和轴5，‘channels_last’模式下则是轴2，3和4
# data_format：字符串，“channels_first”或“channels_last”之一，代表数据的通道维的位置。该参数是Keras 1.x中的image_dim_ordering，“channels_last”对应原本的“tf”，“channels_first”对应原本的“th”。
# 以128x128x128的数据为例，“channels_first”应将数据组织为（3,128,128,128），而“channels_last”应将数据组织为（128,128,128,3）。
# 该参数的默认值是~/.keras/keras.json中设置的值，若从未设置过，则为“channels_last”。
# 输入shape：
# ‘channels_first’模式下，为形如（samples, channels, first_axis_to_pad，first_axis_to_pad, first_axis_to_pad,）的5D张量。
# ‘channels_last’模式下，为形如（samples, first_axis_to_pad，first_axis_to_pad, first_axis_to_pad, channels）的5D张量。

# 输出shape：
# ‘channels_first’模式下，为形如（samples, channels, first_paded_axis，second_paded_axis, third_paded_axis,）的5D张量
# ‘channels_last’模式下，为形如（samples, len_pool_dim1, len_pool_dim2, len_pool_dim3，channels, ）的5D张量

# 3、池化层Pooling
# 3.1、MaxPooling1D层
keras.layers.pooling.MaxPooling1D(pool_size=2, strides=None, padding='valid')

# 对时域1D信号进行最大值池化
# 参数
# pool_size：整数，池化窗口大小
# strides：整数或None，下采样因子，例如设2将会使得输出shape为输入的一半，若为None则默认值为pool_size。
# padding：‘valid’或者‘same’
# 输入shape：形如（samples，steps，features）的3D张量
# 输出shape：形如（samples，downsampled_steps，features）的3D张量

# x = np.random.random(size=(1, 4, 3))
# x
# Out[135]:
# array([[[0.74846466, 0.65318177, 0.61164984],
#         [0.5511416 , 0.38683629, 0.54357629],
#         [0.10375197, 0.68098721, 0.41376335],
#         [0.54915824, 0.5933901 , 0.14964712]]])
# strides=None时的池化结果，output.shape = (x.shape[0], x.shape[1]//pool_size, x.shape[2])
MaxPooling1D(pool_size=2, strides=None, padding='valid')(x)
# Out[136]:
# <tf.Tensor: shape=(1, 2, 3), dtype=float32, numpy=
# array([[[0.74846464, 0.6531818 , 0.6116498 ],
#         [0.5491582 , 0.68098724, 0.41376334]]], dtype=float32)>

# strides=1时的池化结果，output.shape = (x.shape[0], x.shape[1]-pool_size+1, x.shape[2])
MaxPooling1D(pool_size=2, strides=1, padding='valid')(x)
# Out[137]:
# <tf.Tensor: shape=(1, 3, 3), dtype=float32, numpy=
# array([[[0.74846464, 0.6531818 , 0.6116498 ],
#         [0.5511416 , 0.68098724, 0.5435763 ],
#         [0.5491582 , 0.68098724, 0.41376334]]], dtype=float32)>

# 3.2、MaxPooling2D层
keras.layers.pooling.MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)

# 为空域信号施加最大值池化
# 参数
# pool_size：整数或长为2的整数tuple，代表在两个方向（竖直，水平）上的下采样因子，如取（2，2）将使图片在两个维度上均变为原长的一半。为整数意为各个维度值相同且为该数字。
# strides：整数或长为2的整数tuple，或者None，步长值。
# border_mode：‘valid’或者‘same’
# data_format：字符串，“channels_first”或“channels_last”之一，代表图像的通道维的位置。该参数是Keras 1.x中的image_dim_ordering，“channels_last”对应原本的“tf”，“channels_first”对应原本的“th”。
# 以128x128的RGB图像为例，“channels_first”应将数据组织为（3,128,128），而“channels_last”应将数据组织为（128,128,3）。该参数的默认值是~/.keras/keras.json中设置的值，若从未设置过，则为“channels_last”。
# 输入shape
# ‘channels_first’模式下，为形如（samples，channels, rows，cols）的4D张量
# ‘channels_last’模式下，为形如（samples，rows, cols，channels）的4D张量
# 输出shape
# ‘channels_first’模式下，为形如（samples，channels, pooled_rows, pooled_cols）的4D张量
# ‘channels_last’模式下，为形如（samples，pooled_rows, pooled_cols，channels）的4D张量

# 3.3、MaxPooling3D层
keras.layers.pooling.MaxPooling3D(pool_size=(2, 2, 2), strides=None, padding='valid', data_format=None)

# 为3D信号（空域或时空域）施加最大值池化。本层目前只能在使用Theano为后端时可用
# 参数
# pool_size：整数或长为3的整数tuple，代表在三个维度上的下采样因子，如取（2，2，2）将使信号在每个维度都变为原来的一半长。
# strides：整数或长为3的整数tuple，或者None，步长值。
# padding：‘valid’或者‘same’
# data_format：字符串，“channels_first”或“channels_last”之一，代表数据的通道维的位置。该参数是Keras 1.x中的image_dim_ordering，“channels_last”对应原本的“tf”，“channels_first”对应原本的“th”。
# 以128x128x128的数据为例，“channels_first”应将数据组织为（3,128,128,128），而“channels_last”应将数据组织为（128,128,128,3）。
# 该参数的默认值是~/.keras/keras.json中设置的值，若从未设置过，则为“channels_last”。
# 输入shape
# ‘channels_first’模式下，为形如（samples, channels, len_pool_dim1, len_pool_dim2, len_pool_dim3）的5D张量
# ‘channels_last’模式下，为形如（samples, len_pool_dim1, len_pool_dim2, len_pool_dim3，channels, ）的5D张量

# 输出shape
# ‘channels_first’模式下，为形如（samples, channels, pooled_dim1, pooled_dim2, pooled_dim3）的5D张量
# ‘channels_last’模式下，为形如（samples, pooled_dim1, pooled_dim2, pooled_dim3,channels,）的5D张量

# 3.4、AveragePooling1D层
keras.layers.pooling.AveragePooling1D(pool_size=2, strides=None, padding='valid')

# 对时域1D信号进行平均值池化
# 参数
# pool_size：整数，池化窗口大小
# strides：整数或None，下采样因子，例如设2将会使得输出shape为输入的一半，若为None则默认值为pool_size。
# padding：‘valid’或者‘same’
# 输入shape：形如（samples，steps，features）的3D张量
# 输出shape：形如（samples，downsampled_steps，features）的3D张量
# x
# Out[146]:
# array([[[0.67789419, 0.12782174],
#         [0.316845  , 0.43794633],
#         [0.69123408, 0.16167074]]])
# AveragePooling1D(pool_size=2, strides=1, padding='valid')(x)
# Out[147]:
# <tf.Tensor: shape=(1, 2, 2), dtype=float32, numpy=
# array([[[0.4973696 , 0.28288403],
#         [0.5040395 , 0.29980853]]], dtype=float32)>

# 3.5、AveragePooling2D层
keras.layers.pooling.AveragePooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)

# 为空域信号施加平均值池化
# 参数
# pool_size：整数或长为2的整数tuple，代表在两个方向（竖直，水平）上的下采样因子，如取（2，2）将使图片在两个维度上均变为原长的一半。为整数意为各个维度值相同且为该数字。
# strides：整数或长为2的整数tuple，或者None，步长值。
# border_mode：‘valid’或者‘same’
# data_format：字符串，“channels_first”或“channels_last”之一，代表图像的通道维的位置。该参数是Keras 1.x中的image_dim_ordering，“channels_last”对应原本的“tf”，“channels_first”对应原本的“th”。以128x128的RGB图像为例，“channels_first”应将数据组织为（3,128,128），而“channels_last”应将数据组织为（128,128,3）。该参数的默认值是~/.keras/keras.json中设置的值，若从未设置过，则为“channels_last”。
# 输入shape
# ‘channels_first’模式下，为形如（samples，channels, rows，cols）的4D张量。
# ‘channels_last’模式下，为形如（samples，rows, cols，channels）的4D张量。

# 输出shape
# ‘channels_first’模式下，为形如（samples，channels, pooled_rows, pooled_cols）的4D张量。
# ‘channels_last’模式下，为形如（samples，pooled_rows, pooled_cols，channels）的4D张量。

# 3.6、AveragePooling3D层
keras.layers.pooling.AveragePooling3D(pool_size=(2, 2, 2), strides=None, padding='valid', data_format=None)

# 为3D信号（空域或时空域）施加平均值池化。本层目前只能在使用Theano为后端时可用
# 参数
# pool_size：整数或长为3的整数tuple，代表在三个维度上的下采样因子，如取（2，2，2）将使信号在每个维度都变为原来的一半长。
# strides：整数或长为3的整数tuple，或者None，步长值。
# padding：‘valid’或者‘same’
# data_format：字符串，“channels_first”或“channels_last”之一，代表数据的通道维的位置。该参数是Keras 1.x中的image_dim_ordering，“channels_last”对应原本的“tf”，“channels_first”对应原本的“th”。
# 以128x128x128的数据为例，“channels_first”应将数据组织为（3,128,128,128），而“channels_last”应将数据组织为（128,128,128,3）。该参数的默认值是~/.keras/keras.json中设置的值，若从未设置过，则为“channels_last”。
# 输入shape：
# ‘channels_first’模式下，为形如（samples, channels, len_pool_dim1, len_pool_dim2, len_pool_dim3）的5D张量
# ‘channels_last’模式下，为形如（samples, len_pool_dim1, len_pool_dim2, len_pool_dim3，channels, ）的5D张量
# 输出shape：
# ‘channels_first’模式下，为形如（samples, channels, pooled_dim1, pooled_dim2, pooled_dim3）的5D张量
# ‘channels_last’模式下，为形如（samples, pooled_dim1, pooled_dim2, pooled_dim3,channels,）的5D张量

# 3.7、GlobalMaxPooling1D层
keras.layers.pooling.GlobalMaxPooling1D()

# 对于时间信号的全局最大池化
# 输入shape：形如（samples，steps，features）的3D张量。
# 输出shape：形如(samples, features)的2D张量。
# x
# Out[148]:
# array([[[0.67789419, 0.12782174],
#         [0.316845  , 0.43794633],
#         [0.69123408, 0.16167074]]])
# GlobalMaxPooling1D()(x)
# Out[149]: <tf.Tensor: shape=(1, 2), dtype=float32, numpy=array([[0.6912341 , 0.43794632]], dtype=float32)>

# 3.8、GlobalAveragePooling1D层
keras.layers.pooling.GlobalAveragePooling1D()

# 为时域信号施加全局平均值池化
# 输入shape：形如（samples，steps，features）的3D张量
# 输出shape：形如(samples, features)的2D张量

# 3.9、GlobalMaxPooling2D层
keras.layers.pooling.GlobalMaxPooling2D(dim_ordering='default')

# 为空域信号施加全局最大值池化
# 参数
# data_format：字符串，“channels_first”或“channels_last”之一，代表图像的通道维的位置。该参数是Keras
# 1.x中的image_dim_ordering，“channels_last”对应原本的“tf”，“channels_first”对应原本的“th”。
# 以128x128的RGB图像为例，“channels_first”应将数据组织为（3,128,128），而“channels_last”应将数据组织为（128,128,3）。该参数的默认值是~/.keras/keras.json中设置的值，若从未设置过，则为“channels_last”。
# 输入shape：
# ‘channels_first’模式下，为形如（samples，channels, rows，cols）的4D张量
# ‘channels_last’模式下，为形如（samples，rows, cols，channels）的4D张量
# 输出shape：形如(nb_samples, channels)的2D张量

# 3.10、GlobalAveragePooling2D层
keras.layers.pooling.GlobalAveragePooling2D(dim_ordering='default')

# 为空域信号施加全局平均值池化
# 参数
# data_format：字符串，“channels_first”或“channels_last”之一，代表图像的通道维的位置。该参数是Keras
# 1.x中的image_dim_ordering，“channels_last”对应原本的“tf”，“channels_first”对应原本的“th”。
# 以128x128的RGB图像为例，“channels_first”应将数据组织为（3,128,128），而“channels_last”应将数据组织为（128,128,3）。该参数的默认值是~/.keras/keras.json中设置的值，若从未设置过，则为“channels_last”。
# 输入shape：
# ‘channels_first’模式下，为形如（samples，channels, rows，cols）的4D张量
# ‘channels_last’模式下，为形如（samples，rows, cols，channels）的4D张量
# 输出shape：形如(nb_samples, channels)的2D张量

# 2.4.2、全连接RNN网络
keras.layers.recurrent.SimpleRNN(output_dim, init='glorot_uniform', inner_init='orthogonal', activation='tanh', W_regularizer=None, U_regularizer=None,
                                 b_regularizer=None, dropout_W=0.0, dropout_U=0.0)

# inner_init：内部单元的初始化方法
# dropout_W：0~1之间的浮点数，控制输入单元到输入门的连接断开比例
# dropout_U：0~1之间的浮点数，控制输入单元到递归连接的断开比例

# 2.4.3、LSTM层
keras.layers.recurrent.LSTM(output_dim, init='glorot_uniform', inner_init='orthogonal', forget_bias_init='one', activation='tanh', inner_activation='hard_sigmoid',
                            W_regularizer=None, U_regularizer=None, b_regularizer=None, dropout_W=0.0, dropout_U=0.0)

# return_sequences：True返回整个序列，false返回输出序列的最后一个输出
# forget_bias_init：遗忘门偏置的初始化函数，Jozefowicz et al.建议初始化为全1元素
# inner_activation：内部单元激活函数


# 2.5 Embedding层
keras.layers.embeddings.Embedding(input_dim, output_dim, init='uniform', input_length=None, W_regularizer=None, activity_regularizer=None,
                                  W_constraint=None, mask_zero=False, weights=None, dropout=0.0)

# 只能作为模型第一层
# mask_zero：布尔值，确定是否将输入中的‘0’看作是应该被忽略的‘填充’（padding）值，该参数在使用递归层处理变长输入时有用。设置为True的话，模型中后续的层必须都支持masking，否则会抛出异常

# merge层
# Concatenate层
x = np.random.random(size= (1, 4, 2))
y = np.random.random(size= (1, 2, 2))
Concatenate(axis=1)([x, y])
# Out[155]:
# <tf.Tensor: shape=(1, 6, 2), dtype=float32, numpy=
# array([[[0.08389397, 0.96558064],
#         [0.55055165, 0.14813688],
#         [0.8869185 , 0.3465255 ],
#         [0.86723596, 0.95987594],
#         [0.04708383, 0.8001296 ],
#         [0.0020749 , 0.02442509]]], dtype=float32)>

# Add层：
# x
# Out[159]:
# array([[[0.24305341, 0.35842983],
#         [0.63126718, 0.29898466]]])
# y
# Out[160]:
# array([[[0.04708383, 0.80012958],
#         [0.0020749 , 0.02442509]]])
Add()([x, y])
# Out[161]:
# <tf.Tensor: shape=(1, 2, 2), dtype=float32, numpy=
# array([[[0.29013723, 1.1585594 ],
#         [0.6333421 , 0.32340974]]], dtype=float32)>

# 标准化层
BatchNormalization(axis=-1,  momentum=0.99,  epsilon=1e-3, center=True,  scale=True, beta_initializer='zeros',
               gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones',
               beta_regularizer=None,  gamma_regularizer=None, beta_constraint=None, gamma_constraint=None,)
# 用keras的BN时切记要设置training=False
# keras 的BN层的call函数里面有个默认参数traing， 默认是None。此参数意义如下：
# training=False/0, 训练时通过每个batch的移动平均的均值、方差去做批归一化，测试时拿整个训练集的均值、方差做归一化
# training=True/1/None，训练时通过当前batch的均值、方差去做批归一化，测试时拿整个训练集的均值、方差做归一化
# 当training=None时，训练和测试的批归一化方式不一致，导致validation的输出指标翻车。
# 当training=True时，拿训练完的模型预测一个样本和预测一个batch的样本的差异非常大，也就是预测的结果根据batch的大小会不同！导致模型结果无法准确评估！

# 该层在每个batch上将前一层的激活值重新规范化，即使得其输出数据的均值接近0，其标准差接近1
# 参数
# epsilon：大于0的小浮点数，用于防止除0错误
# mode：整数，指定规范化的模式，取0或1
# 0：按特征规范化，输入的各个特征图将独立被规范化。规范化的轴由参数axis指定。注意，如果输入是形如（samples，channels，rows，cols）的4D图像张量，则应设置规范化的轴为1，即沿着通道轴规范化。输入格式是‘tf’同理。
# 1：按样本规范化，该模式默认输入为2D
# axis：整数，指定当mode=0时规范化的轴。例如输入是形如（samples，channels，rows，cols）的4D图像张量，则应设置规范化的轴为1，意味着对每个特征图进行规范化
# momentum：在按特征规范化时，计算数据的指数平均数和标准差时的动量
# weights：初始化权重，为包含2个numpy array的list，其shape为[(input_shape,),(input_shape)]
# beta_init：beta的初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的Theano函数。该参数仅在不传递weights参数时有意义。
# gamma_init：gamma的初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的Theano函数。该参数仅在不传递weights参数时有意义。
# 输入shape
# 任意，当使用本层为模型首层时，指定input_shape参数时有意义。
#
# 输出shape
# 与输入shape相同

x = Dense(32, activation='relu')(x)
x = BatchNormalization()(x)
# x
# Out[164]:
# array([[[0.98303387, 0.72000026],
#         [0.25455523, 0.69712237],
#         [0.4964275 , 0.2443441 ]]])
# BatchNormalization()(x)
# Out[165]:
# <tf.Tensor: shape=(1, 3, 2), dtype=float32, numpy=
# array([[[0.98254275, 0.71964055],
#         [0.25442806, 0.6967741 ],
#         [0.4961795 , 0.24422203]]], dtype=float32)>

# 噪声层Noise
# GaussianNoise层
keras.layers.noise.GaussianNoise(sigma)

# 为层的输入施加0均值，标准差为sigma的加性高斯噪声。该层在克服过拟合时比较有用，你可以将它看作是随机的数据提升。高斯噪声是需要对输入数据进行破坏时的自然选择。
# 一个使用噪声层的典型案例是构建去噪自动编码器，即Denoising AutoEncoder（DAE）。该编码器试图从加噪的输入中重构无噪信号，以学习到原始信号的鲁棒性表示。
# 因为这是一个起正则化作用的层，该层只在训练时才有效。
# 参数
# sigma：浮点数，代表要产生的高斯噪声标准差
# 输入shape, 任意，当使用该层为模型首层时需指定input_shape参数
# 输出shape与输入相同

# GaussianDropout层
keras.layers.noise.GaussianDropout(p)

# 为层的输入施加以1为均值，标准差为sqrt(p/(1-p)的乘性高斯噪声
# 因为这是一个起正则化作用的层，该层只在训练时才有效。
# 参数
# p：浮点数，断连概率，与Dropout层相同
# 输入shape, 任意，当使用该层为模型首层时需指定input_shape参数
# 输出shape, 与输入相同

# 点乘注意力层
tf.keras.layers.Attention(use_scale=False, score_mode='dot', **kwargs)

# 参数
# use_scale	如果为 True, 将会创建一个标量的变量对注意力分数进行缩放.
# causal	Boolean. 可以设置为 True 用于解码器的自注意力. 它会添加一个mask, 使位置i 看不到未来的信息.
# dropout	0到1之间的浮点数. 对注意力分数的dropout
# 调用参数:
#
# inputs:
# query:  [batch_size, Tq, dim]
# value: [batch_size, Tv, dim]
# key: [batch_size, Tv, dim], 如果没有给定, 则默认key=value
# mask:
#
# query_mask: [batch_size, Tq], 如果给定, mask==False的位置输出为0.
# value_mask: [batch_size, Tv], 如果给定, mask==False的位置不会对输出产生贡献.

att = Attention(use_scale=False, score_mode='dot',)
# x
# Out[186]:
# array([[[0.93258257, 0.31132519],
#         [0.52446679, 0.60867065],
#         [0.16103757, 0.40487986]]])
att([x, x])
# Out[187]:
# <tf.Tensor: shape=(1, 3, 2), dtype=float32, numpy=
# array([[[0.624818  , 0.43119362],
#         [0.5810993 , 0.44362986],
#         [0.54960877, 0.44580445]]], dtype=float32)>

# 计算的步骤如下:
# 1、计算点乘注意力分数[batch_size, Tq, Tv]: scores = tf.matmul(query, key, transpose_b=True)
# 2、计算softmax: distribution = tf.nn.softmax(scores)
# 3、对value加权求和: tf.matmul(distribution, value), 得到shape为[batch_size, Tq, dim]的输出.

tf.matmul(tf.nn.softmax(tf.matmul(x, x, transpose_b=True)), x)
# Out[190]:
# <tf.Tensor: shape=(1, 3, 2), dtype=float64, numpy=
# array([[[0.62481802, 0.43119359],
#         [0.58109926, 0.44362988],
#         [0.5496088 , 0.44580443]]])>


# MultiHeadAttention 层
tf.keras.layers.MultiHeadAttention(num_heads, key_dim, value_dim=None, dropout=0.0, use_bias=True, output_shape=None, attention_axes=None,
    kernel_initializer="glorot_uniform", bias_initializer="zeros", kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
    kernel_constraint=None, bias_constraint=None, **kwargs )
# 如果 query 、 key, value 相同，那么这就是self-attention。 query 中的每个时间步都关注 key 中的相应序列，并返回一个固定宽度的向量。
# 该层首先投影 query 、 key 和 value 。这些（实际上）是长度为 num_attention_heads 的张量列表，其中对应的形状是
# (batch_size, <query dimensions>, key_dim) , (batch_size, <key/value dimensions>, key_dim) , (batch_size, <key/value dimensions>, value_dim) 。
# 然后,查询和关键张量被点积和缩放。这些被软放大以获得注意力概率。然后用这些概率对价值张量进行插值,再串联成一个单一的张量。
# 最后,以最后一个维度为value_dim的结果张量可以采取线性投影并返回。
# 参数：
# num_heads	注意力头的数量。
# key_dim	查询和键的每个注意头的大小。
# value_dim	价值的每个注意头的大小。
# dropout	Dropout probability.
# use_bias	布尔值,密集层是否使用偏置向量/矩阵。
# output_shape	输出张量的预期形状,除了批次和序列的dim。如果没有指定,则投射回关键特征的dim。
# attention_axes	应用注意力的轴。 None 意味着关注所有轴，而是批处理、头部和特征。
# kernel_initializer	密集层内核的初始化器。
# bias_initializer	密集层偏差的初始化器。
# kernel_regularizer	密集层核的正则器。
# bias_regularizer	密集层偏差的正则器。
# activity_regularizer	密集层活动的正则器。
# kernel_constraint	密集层果核的约束条件。
# bias_constraint	密集层果核的约束条件。
# Call arguments:
# query : 查询形状 (B, T, dim) Tensor 。
# value :形状 (B, S, dim) Tensor 。
# key :形状 (B, S, dim) 的可选键 Tensor 。如果没有给出，将对 key 和 value 使用value key 这 value value 常见的情况。
# attention_mask ：形状为 (B, T, S) 的布尔掩码，可防止对某些位置的注意。布尔掩码指定哪些查询元素可以关注哪些关键元素，1 表示关注，0 表示不关注。对于缺少的批次维度和头部维度，可能会发生广播。
# return_attention_scores ：一个布尔值，指示输出是否应为 (attention_output, attention_scores) 如果 True ，或者 attention_output 如果 False 。默认为 False 。
# training ：Python 布尔值，指示该层应在训练模式（添加 dropout）还是推理模式（无 dropout）下运行。默认使用父层/模型的训练模式，如果没有父层，则默认为 False（推理）。
# Returns
# attention_output	计算结果，形状为 (B, T, E) ，其中 T 表示目标序列形状，如果 output_shape 为 None ,则 E 是查询输入的最后一维。否则，多头输出将投影到 output_shape 指定的形状。
# attention_scores	[可选] 注意力轴上的多头注意力系数。

# AdditiveAttention层：
# 加法注意力层
tf.keras.layers.AdditiveAttention(
    use_scale=True, **kwargs
)

# 计算的步骤如下:
# 1、把query和value的shape分别转换成[batch_size, Tq, 1, dim]和[batch_size, 1, Tv, dim]
# 2、计算注意力分数[batch_size, Tq, Tv]: scores = tf.reduce_sum(scale * tf.tanh(query + value), axis=-1) ,
# 若use_scale=True这里scale 是一个可训练的标量，会对注意力分数进行缩放; 若use_scale=Flase, 则scale=1
# 3、进行softmax: distribution = tf.nn.softmax(scores)
# 4、对value加权求和: tf.matmul(distribution, value), 得到shape为batch_size, Tq, dim]的输出

# x
# Out[209]:
# array([[[0.93258257, 0.31132519],
#         [0.52446679, 0.60867065],
#         [0.16103757, 0.40487986]]])
# add_att = AdditiveAttention()
# add_att([x, x])
# Out[211]:
# <tf.Tensor: shape=(1, 3, 2), dtype=float32, numpy=
# array([[[0.5527128 , 0.43975556],
#         [0.56397974, 0.43900397],
#         [0.57884556, 0.43676037]]], dtype=float32)>
# add_att.weights
# Out[212]: [<tf.Variable 'additive_attention_4/scale:0' shape=(2,) dtype=float32, numpy=array([ 0.62221444, -0.11633062], dtype=float32)>]
tf.matmul(tf.nn.softmax(tf.reduce_sum(add_att.weights[0].numpy() * tf.tanh(Reshape(target_shape=(1, 3, 1, 2))(x) + Reshape(target_shape=(1, 1, 3, 2))(x)), axis=-1)), x)
# <tf.Tensor: shape=(1, 1, 3, 2), dtype=float32, numpy=
# array([[[[0.5527128 , 0.43975556],
#          [0.56397974, 0.43900397],
#          [0.57884556, 0.43676037]]]], dtype=float32)>

# 输入是 query 形状的张量 [batch_size, Tq, dim] ， value 形状的张量 [batch_size, Tv, dim] 和 key 形状的张量 [batch_size, Tv, dim] 。计算遵循以下步骤：
#
# query 和 key 分别重塑为形状 [batch_size, Tq, 1, dim] 和 [batch_size, 1, Tv, dim] 。
# 计算形状为 [batch_size, Tq, Tv] 的分数作为非线性总和： scores = tf.reduce_sum(tf.tanh(query + key), axis=-1)
# 使用分数计算形状为 [batch_size, Tq, Tv] distribution = tf.nn.softmax(scores) ：distribution = tf.nn.softmax（scores）。
# 使用 distribution 创建形状为 [batch_size, Tq, dim] 的 value 的线性组合： return tf.matmul(distribution, value) 。
# Args
# use_scale	如果为 True ，将创建一个变量来缩放注意力得分。
# causal	布尔值。为解码器自注意力设置为 True 。添加一个掩码，使位置 i 无法处理位置 j > i 。这可以防止信息从未来流向过去。默认为 False 。
# dropout	在0和1之间的浮动值。 注意力分数要下降的单位的分数。默认为0.0。
# Call Args:
# inputs ：以下张量的列表：
# 查询：查询形状为 [batch_size, Tq, dim] Tensor 。
# value：形状 [batch_size, Tv, dim] 值 Tensor 。
# key：形状为 [batch_size, Tv, dim] 可选密钥 Tensor 。如果未给出，则将 value 用作 key 和 value ，这是最常见的情况。
# mask ：以下张量的列表：
# query_mask：形状为 [batch_size, Tq] 布尔掩码 Tensor 。如果给定，则在 mask==False 的位置输出为零。
# value_mask： [batch_size, Tv] 形状的布尔掩码 Tensor 。如果给定，将应用遮罩，以使 mask==False 位置上的值不影响结果。
# training ：Python布尔值，指示该层是应在训练模式下（添加退出）还是在推理模式下（不退出）运行。
# return_attention_scores : bool, it True ，返回注意力分数（在屏蔽和 softmax 之后）作为额外的输出参数。
# Output:
# 形状 [batch_size, Tq, dim] 注意力输出。[可选] 使用形状 [batch_size, Tq, Tv] 进行掩蔽和 softmax 后的注意力分数。
#
# query ， value 和 key 的含义取决于应用程序。例如，在文本相似的情况下， query 是第一条文本的序列嵌入，而 value 是第二条文本的序列嵌入。 key 通常是与 value 相同的张量。

def main():
    pass


if __name__ == '__main__':
    main()
