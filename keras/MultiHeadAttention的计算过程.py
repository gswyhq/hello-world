
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers.attention.multi_head_attention import activation, MultiHeadAttention

x = np.array([[[8., 9., 5.],
        [7., 4., 5.]]], dtype='float32')
key_dim=2
mha=MultiHeadAttention(num_heads=2, key_dim=2)
mha(x, x)
# Out[19]:
# <tf.Tensor: shape=(1, 2, 3), dtype=float32, numpy=
# array([[[-3.5109825, -9.086956 , -2.1889913],
#         [-3.351711 , -8.958122 , -2.1655416]]], dtype=float32)>

mha.weights
# Out[21]:
# [ < tf.Variable
# 'multi_head_attention_6/query/kernel:0'
# shape = (3, 2, 2)
# dtype = float32, numpy =
# array([[[-0.46178007, 0.3666125],
#         [-0.41106933, 0.1738016]],
#
#        [[0.68113333, 0.6252021],
#         [0.17564464, -0.6051753]],
#
#        [[0.18035936, -0.16819203],
#         [-0.1765936, 0.62256664]]], dtype=float32) >,
# < tf.Variable
# 'multi_head_attention_6/query/bias:0'
# shape = (2, 2)
# dtype = float32, numpy =
# array([[0., 0.],
#        [0., 0.]], dtype=float32) >,
# < tf.Variable
# 'multi_head_attention_6/key/kernel:0'
# shape = (3, 2, 2)
# dtype = float32, numpy =
# array([[[-0.1812095, -0.2381671],
#         [-0.3108195, -0.19722617]],
#
#        [[0.1958372, 0.26021433],
#         [-0.4131718, -0.4558657]],
#
#        [[-0.12985998, 0.09231877],
#         [0.34172648, 0.43733758]]], dtype=float32) >,
# < tf.Variable
# 'multi_head_attention_6/key/bias:0'
# shape = (2, 2)
# dtype = float32, numpy =
# array([[0., 0.],
#        [0., 0.]], dtype=float32) >,
# < tf.Variable
# 'multi_head_attention_6/value/kernel:0'
# shape = (3, 2, 2)
# dtype = float32, numpy =
# array([[[0.6978424, 0.06613141],
#         [-0.04118121, -0.44537315]],
#
#        [[-0.08198339, -0.5837982],
#         [0.37101918, -0.4676124]],
#
#        [[0.55096287, 0.4578628],
#         [0.6616655, -0.45340028]]], dtype=float32) >,
# < tf.Variable
# 'multi_head_attention_6/value/bias:0'
# shape = (2, 2)
# dtype = float32, numpy =
# array([[0., 0.],
#        [0., 0.]], dtype=float32) >,
# < tf.Variable
# 'multi_head_attention_6/attention_output/kernel:0'
# shape = (2, 2, 3)
# dtype = float32, numpy =
# array([[[0.07295823, -0.7614871, -0.21277499],
#         [0.49636447, 0.20486355, -0.6798671]],
#
#        [[-0.45434523, 0.63044345, 0.35442042],
#         [-0.00114834, 0.67602, 0.4450711]]], dtype=float32) >,
# < tf.Variable
# 'multi_head_attention_6/attention_output/bias:0'
# shape = (3,)
# dtype = float32, numpy = array([0., 0., 0.], dtype=float32) >]

q_k = mha.weights[0] # shape=(x.shape[-1], num_heads, key_dim)
k_k = mha.weights[2]
v_k = mha.weights[4]
att_k = mha.weights[6]

query = mha._query_dense(x)
key = mha._key_dense(x)
value = mha._value_dense(x)

# 计算过程：
# 1、每个Head对应三个权重矩阵用于从输入向量中计算(Q, K, V)
# 2、每个Head根据自己的(Q, K, V)根据Attention过程计算得出Attention value向量。n个Head一共有n个Attention value。
# 3、将这n个Attention value向量连接起来，乘以一个权重矩阵，以使其转变为与输入向量大小相同的矩阵

# 第一步 key的计算，query, value计算同理
tf.einsum('abc,cde->abde', x, k_k)
# Out[178]:
# <tf.Tensor: shape=(1, 2, 2, 2), dtype=float32, numpy=
# array([[[[-0.3364411 ,  0.89818597],
#          [-4.4964695 , -3.4939127 ]],
#         [[-1.1344175 , -0.16471863],
#          [-2.119791  , -1.0173581 ]]]], dtype=float32)>
key
# Out[179]:
# <tf.Tensor: shape=(1, 2, 2, 2), dtype=float32, numpy=
# array([[[[-0.3364411 ,  0.89818597],
#          [-4.4964695 , -3.4939127 ]],
#         [[-1.1344175 , -0.16471863],
#          [-2.119791  , -1.0173581 ]]]], dtype=float32)>

# 第二步，计算attention
att_s = tf.einsum('aecd,abcd->acbe', key, tf.multiply(query, 1.0/math.sqrt(key_dim)))
att_s = activation.Softmax()(att_s, None)
attention_out = tf.einsum('acbe,aecd->abcd', att_s, value)

# 第三步，计算最后的输出：
tf.einsum('abcd,cde->abe', attention_out, att_k)
# Out[186]:
# <tf.Tensor: shape=(1, 2, 3), dtype=float32, numpy=
# array([[[-3.5109825, -9.086956 , -2.1889913],
#         [-3.351711 , -8.958122 , -2.1655416]]], dtype=float32)>
mha(x, x)
# Out[187]:
# <tf.Tensor: shape=(1, 2, 3), dtype=float32, numpy=
# array([[[-3.5109825, -9.086956 , -2.1889913],
#         [-3.351711 , -8.958122 , -2.1655416]]], dtype=float32)>

# attention的具体运算的过程是这样的：
# 针对句子中的某一个单词W，计算其余单词以及它自身所对应的权重，由此可以知道这个句子当中每一个单词对目标单词的重要性。
# 最后将这些单词按照权重的softmax加权相加，代表原来的目标单词W的向量。
# 对句子中某一个单词，计算他的权重的过程是这样的：
# 首先，用这个单词的query向量和句子中其余单词的key向量挨个相乘，对应每一个单词都得到一个值，这就是初始的权值。
# 然后，用用这个权值除以根号下key向量的长度，即将这个权值放缩，按照“Attention is all you need”的说法，softmax里面的值太大会导致梯度为零。
# 接下来就是将放缩以后的权值用softmax归一化，这样各个word所对应的权值加起来等于一。
# 最后将各个word的value向量按照权重进行加权求和，来代表这个单词本身的词向量。（可以看出，这里的value从某种意义上代表了原单词的embedding向量。）

