#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 加载 https://github.com/ZhuiyiTechnology/pretrained-models 模型，需要tensorflow 1.14, 而常用tensorflow2.x 加载失败；
# 故而从头构建SimBERT模型；

import os
import numpy as np
import keras
import tensorflow as tf
from keras import backend as K
from keras import initializers, activations
from keras.engine.input_layer import InputLayer
from keras_bert.layers.embedding import TokenEmbedding
from keras.layers import Embedding, Add, Dropout, Input, Dense, Layer, InputSpec, Lambda
from keras_pos_embd.pos_embd import PositionEmbedding
# from keras_layer_normalization.layer_normalization import LayerNormalization
# from keras_multi_head.multi_head_attention import MultiHeadAttention
# from keras_position_wise_feed_forward.feed_forward import FeedForward
from keras.models import Model, load_model

from distutils.util import strtobool
from sklearn.metrics.pairwise import cosine_similarity

# tf.__version__
# Out[5]: '2.9.1'
# keras.__version__
# Out[6]: '2.9.0'

# 判断是否启用重计算（通过时间换空间）
do_recompute = strtobool(os.environ.get('RECOMPUTE', '0'))

def gelu_erf(x):
    """基于Erf直接计算的gelu函数
    """
    return 0.5 * x * (1.0 + tf.math.erf(x / np.sqrt(2.0)))


def align(tensor, axes, ndim=None):
    """重新对齐tensor（批量版expand_dims）
    axes：原来的第i维对齐新tensor的第axes[i]维；
    ndim：新tensor的维度。
    """
    assert len(axes) == K.ndim(tensor)
    assert ndim or min(axes) >= 0
    ndim = ndim or max(axes) + 1
    indices = [None] * ndim
    for i in axes:
        indices[i] = slice(None)
    return tensor[indices]

def K_flatten(tensor, start=None, end=None):
    """将tensor从start到end的维度展平
    """
    start, end = start or 0, end or K.ndim(tensor)
    shape = K.shape(tensor)
    shape = [s or shape[i] for i, s in enumerate(K.int_shape(tensor))]
    shape = shape[:start] + [K.prod(shape[start:end])] + shape[end:]
    return K.reshape(tensor, shape)

def is_string(s):
    """判断是否是字符串
    """
    return isinstance(s, str)


def attention_normalize(a, axis=-1, method='softmax'):
    """不同的注意力归一化方案
    softmax：常规/标准的指数归一化；
    squared_relu：来自 https://arxiv.org/abs/2202.10447 ；
    softmax_plus：来自 https://kexue.fm/archives/8823 。
    """
    if method == 'softmax':
        return K.softmax(a, axis=axis)
    else:
        mask = K.cast(a > -1e12 / 10, K.floatx())
        l = K.maximum(K.sum(mask, axis=axis, keepdims=True), 1)
        if method == 'squared_relu':
            return K.relu(a)**2 / l
        elif method == 'softmax_plus':
            scale = K.log(l) / np.log(512) * mask + 1 - mask
            return K.softmax(a * scale, axis=axis)
    return a

def sequence_masking(x, mask, value=0, axis=None):
    """为序列条件mask的函数
    mask: 形如(batch_size, seq_len)的0-1矩阵；
    value: mask部分要被替换成的值，可以是'-inf'或'inf'；
    axis: 序列所在轴，默认为1；
    """
    if mask is None:
        return x
    else:
        x_dtype = K.dtype(x)
        if x_dtype == 'bool':
            x = K.cast(x, 'int32')
        if K.dtype(mask) != K.dtype(x):
            mask = K.cast(mask, K.dtype(x))
        if value == '-inf':
            value = -1e12
        elif value == 'inf':
            value = 1e12
        if axis is None:
            axis = 1
        elif axis < 0:
            axis = K.ndim(x) + axis
        assert axis > 0, 'axis must be greater than 0'
        mask = align(mask, [0, axis], K.ndim(x))
        value = K.cast(value, K.dtype(x))
        x = x * mask + value * (1 - mask)
        if x_dtype == 'bool':
            x = K.cast(x, 'bool')
        return x

def recompute_grad(call):
    """重计算装饰器（用来装饰Keras层的call函数）
    关于重计算，请参考：https://arxiv.org/abs/1604.06174
    """
    if not do_recompute:
        return call

    def inner(self, inputs, **kwargs):
        """定义需要求梯度的函数以及重新定义求梯度过程
        （参考自官方自带的tf.recompute_grad函数）
        """
        flat_inputs = nest.flatten(inputs)
        call_args = tf_inspect.getfullargspec(call).args
        for key in ['mask', 'training']:
            if key not in call_args and key in kwargs:
                del kwargs[key]

        def kernel_call():
            """定义前向计算
            """
            return call(self, inputs, **kwargs)

        def call_and_grad(*inputs):
            """定义前向计算和反向计算
            """
            if is_tf_keras:
                with tape.stop_recording():
                    outputs = kernel_call()
                    outputs = tf.identity(outputs)
            else:
                outputs = kernel_call()

            def grad_fn(doutputs, variables=None):
                watches = list(inputs)
                if variables is not None:
                    watches += list(variables)
                with tf.GradientTape() as t:
                    t.watch(watches)
                    with tf.control_dependencies([doutputs]):
                        outputs = kernel_call()
                grads = t.gradient(
                    outputs, watches, output_gradients=[doutputs]
                )
                del t
                return grads[:len(inputs)], grads[len(inputs):]

            return outputs, grad_fn

        if is_tf_keras:  # 仅在tf >= 2.0下可用
            outputs, grad_fn = call_and_grad(*flat_inputs)
            flat_outputs = nest.flatten(outputs)

            def actual_grad_fn(*doutputs):
                grads = grad_fn(*doutputs, variables=self.trainable_weights)
                return grads[0] + grads[1]

            watches = flat_inputs + self.trainable_weights
            watches = [tf.convert_to_tensor(x) for x in watches]
            tape.record_operation(
                call.__name__, flat_outputs, watches, actual_grad_fn
            )
            return outputs
        else:  # keras + tf >= 1.14 均可用
            return graph_mode_decorator(call_and_grad, *flat_inputs)

    return inner

def integerize_shape(func):
    """装饰器，保证input_shape一定是int或None
    """
    def convert(item):
        if hasattr(item, '__iter__'):
            return [convert(i) for i in item]
        elif hasattr(item, 'value'):
            return item.value
        else:
            return item

    def new_func(self, input_shape):
        input_shape = convert(input_shape)
        return func(self, input_shape)

    return new_func

class ScaleOffset(Layer):
    """简单的仿射变换层（最后一维乘上gamma向量并加上beta向量）
    说明：1、具体操作为最后一维乘上gamma向量并加上beta向量；
         2、如果直接指定scale和offset，那么直接常数缩放和平移；
         3、hidden_*系列参数仅为有条件输入时(conditional=True)使用，
            用于通过外部条件控制beta和gamma。
    """
    def __init__(
        self,
        scale=True,
        offset=True,
        conditional=False,
        hidden_units=None,
        hidden_activation='linear',
        hidden_initializer='glorot_uniform',
        **kwargs
    ):
        super(ScaleOffset, self).__init__(**kwargs)
        self.scale = scale
        self.offset = offset
        self.conditional = conditional
        self.hidden_units = hidden_units
        self.hidden_activation = activations.get(hidden_activation)
        self.hidden_initializer = initializers.get(hidden_initializer)

    @integerize_shape
    def build(self, input_shape):
        super(ScaleOffset, self).build(input_shape)

        if self.conditional:
            input_shape = input_shape[0]

        if self.offset is True:
            self.beta = self.add_weight(
                name='beta', shape=(input_shape[-1],), initializer='zeros'
            )
        if self.scale is True:
            self.gamma = self.add_weight(
                name='gamma', shape=(input_shape[-1],), initializer='ones'
            )

        if self.conditional:

            if self.hidden_units is not None:
                self.hidden_dense = Dense(
                    units=self.hidden_units,
                    activation=self.hidden_activation,
                    use_bias=False,
                    kernel_initializer=self.hidden_initializer
                )

            if self.offset is not False and self.offset is not None:
                self.beta_dense = Dense(
                    units=input_shape[-1],
                    use_bias=False,
                    kernel_initializer='zeros'
                )
            if self.scale is not False and self.scale is not None:
                self.gamma_dense = Dense(
                    units=input_shape[-1],
                    use_bias=False,
                    kernel_initializer='zeros'
                )

    def compute_mask(self, inputs, mask=None):
        if self.conditional:
            return mask if mask is None else mask[0]
        else:
            return mask

    @recompute_grad
    def call(self, inputs):
        """如果带有条件，则默认以list为输入，第二个是条件
        """
        if self.conditional:
            inputs, conds = inputs
            if self.hidden_units is not None:
                conds = self.hidden_dense(conds)
            conds = align(conds, [0, -1], K.ndim(inputs))

        if self.scale is not False and self.scale is not None:
            gamma = self.gamma if self.scale is True else self.scale
            if self.conditional:
                gamma = gamma + self.gamma_dense(conds)
            inputs = inputs * gamma

        if self.offset is not False and self.offset is not None:
            beta = self.beta if self.offset is True else self.offset
            if self.conditional:
                beta = beta + self.beta_dense(conds)
            inputs = inputs + beta

        return inputs

    def compute_output_shape(self, input_shape):
        if self.conditional:
            return input_shape[0]
        else:
            return input_shape

    def get_config(self):
        config = {
            'scale': self.scale,
            'offset': self.offset,
            'conditional': self.conditional,
            'hidden_units': self.hidden_units,
            'hidden_activation': activations.serialize(self.hidden_activation),
            'hidden_initializer':
                initializers.serialize(self.hidden_initializer),
        }
        base_config = super(ScaleOffset, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



class LayerNormalization(ScaleOffset):
    """(Conditional) Layer Normalization
    """
    def __init__(
        self, zero_mean=True, unit_variance=True, epsilon=None, offset=True, **kwargs
    ):
        super(LayerNormalization, self).__init__(**kwargs)
        self.zero_mean = zero_mean
        self.unit_variance = unit_variance
        self.epsilon = epsilon or K.epsilon()

    @recompute_grad
    def call(self, inputs):
        """如果是条件Layer Norm，则默认以list为输入，第二个是条件
        """
        if self.conditional:
            inputs, conds = inputs

        if self.zero_mean:
            mean = K.mean(inputs, axis=-1, keepdims=True)
            inputs = inputs - mean
        if self.unit_variance:
            variance = K.mean(K.square(inputs), axis=-1, keepdims=True)
            inputs = inputs / K.sqrt(variance + self.epsilon)

        if self.conditional:
            inputs = [inputs, conds]

        return super(LayerNormalization, self).call(inputs)

    def get_config(self):
        config = {
            'zero_mean': self.zero_mean,
            'unit_variance': self.unit_variance,
            'epsilon': self.epsilon,
        }
        base_config = super(LayerNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class FeedForward(Layer):
    """FeedForward层
    如果activation不是一个list，那么它就是两个Dense层的叠加；如果activation是
    一个list，那么第一个Dense层将会被替换成门控线性单元（Gated Linear Unit）。
    参考论文: https://arxiv.org/abs/2002.05202
    """
    def __init__(
        self,
        units,
        activation='relu',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        **kwargs
    ):
        super(FeedForward, self).__init__(**kwargs)
        self.units = units
        if not isinstance(activation, list):
            activation = [activation]
        self.activation = [activations.get(act) for act in activation]
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)

    @integerize_shape
    def build(self, input_shape):
        super(FeedForward, self).build(input_shape)
        output_dim = input_shape[-1]

        for i, activation in enumerate(self.activation):
            i_dense = Dense(
                units=self.units,
                activation=activation,
                use_bias=self.use_bias,
                kernel_initializer=self.kernel_initializer
            )
            setattr(self, 'i%s_dense' % i, i_dense)

        self.o_dense = Dense(
            units=output_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer
        )

    @recompute_grad
    def call(self, inputs):
        x = self.i0_dense(inputs)
        for i in range(1, len(self.activation)):
            x = x * getattr(self, 'i%s_dense' % i)(inputs)
        x = self.o_dense(x)
        return x

    def get_config(self):
        config = {
            'units': self.units,
            'activation': [
                activations.serialize(act) for act in self.activation
            ],
            'use_bias': self.use_bias,
            'kernel_initializer':
                initializers.serialize(self.kernel_initializer),
        }
        base_config = super(FeedForward, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class MultiHeadAttention(Layer):
    """多头注意力机制
    """
    def __init__(
        self,
        heads,
        head_size,
        out_dim=None,
        key_size=None,
        use_bias=True,
        normalization='softmax',
        attention_scale=True,
        attention_dropout=None,
        return_attention_scores=False,
        kernel_initializer='glorot_uniform',
        **kwargs
    ):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.heads = heads
        self.head_size = head_size
        self.out_dim = out_dim or heads * head_size
        self.key_size = key_size or head_size
        self.use_bias = use_bias
        self.normalization = normalization
        self.attention_scale = attention_scale
        self.attention_dropout = attention_dropout
        self.return_attention_scores = return_attention_scores
        self.kernel_initializer = initializers.get(kernel_initializer)

    def build(self, input_shape):
        super(MultiHeadAttention, self).build(input_shape)
        self.q_dense = Dense(
            units=self.key_size * self.heads,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer
        )
        self.k_dense = Dense(
            units=self.key_size * self.heads,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer
        )
        self.v_dense = Dense(
            units=self.head_size * self.heads,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer
        )
        self.o_dense = Dense(
            units=self.out_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer
        )

    @recompute_grad
    def call(self, inputs, mask=None, **kwargs):
        """实现多头注意力
        q_mask: 对输入的query序列的mask。
                主要是将输出结果的padding部分置0。
        v_mask: 对输入的value序列的mask。
                主要是防止attention读取到padding信息。
        """
        q, k, v = inputs[:3]
        q_mask, v_mask = None, None
        if mask is not None:
            q_mask, v_mask = mask[0], mask[2]
        # 线性变换
        qw = self.q_dense(q)
        kw = self.k_dense(k)
        vw = self.v_dense(v)
        # 形状变换
        qw = K.reshape(qw, (-1, K.shape(q)[1], self.heads, self.key_size))
        kw = K.reshape(kw, (-1, K.shape(k)[1], self.heads, self.key_size))
        vw = K.reshape(vw, (-1, K.shape(v)[1], self.heads, self.head_size))
        # Attention
        qkv_inputs = [qw, kw, vw] + inputs[3:]
        qv_masks = [q_mask, v_mask]
        o, a = self.pay_attention_to(qkv_inputs, qv_masks, **kwargs)
        # print("o.shape", o.shape)
        # 完成输出
        o = self.o_dense(K_flatten(o, 2))
        # 返回结果
        if self.return_attention_scores:
            return [o, a]
        else:
            return o

    def pay_attention_to(self, inputs, mask=None, **kwargs):
        """实现标准的乘性多头注意力
        a_bias: 对attention矩阵的bias。
                不同的attention bias对应不同的应用。
        p_bias: 在attention里的位置偏置。
                一般用来指定相对位置编码的种类。
        说明: 这里单独分离出pay_attention_to函数，是为了方便
              继承此类来定义不同形式的attention；此处要求
              返回o.shape=(batch_size, seq_len, heads, head_size)。
        """
        (qw, kw, vw), n = inputs[:3], 3
        q_mask, v_mask = mask
        a_bias, p_bias = kwargs.get('a_bias'), kwargs.get('p_bias')
        if a_bias:
            a_bias = inputs[n]
            n += 1
        if p_bias == 'rotary':
            qw, kw = apply_rotary_position_embeddings(inputs[n], qw, kw)
        # Attention
        a = tf.einsum('bjhd,bkhd->bhjk', qw, kw)
        # 处理位置编码
        if p_bias == 'typical_relative':
            position_bias = inputs[n]
            a = a + tf.einsum('bjhd,jkd->bhjk', qw, position_bias)
        elif p_bias == 't5_relative':
            position_bias = K.permute_dimensions(inputs[n], (2, 0, 1))
            a = a + K.expand_dims(position_bias, 0)
        # Attention（续）
        if self.attention_scale:
            a = a / self.key_size**0.5
        if a_bias is not None:
            if K.ndim(a_bias) == 3:
                a_bias = align(a_bias, [0, -2, -1], K.ndim(a))
            a = a + a_bias
        a = sequence_masking(a, v_mask, '-inf', -1)
        A = attention_normalize(a, -1, self.normalization)
        if self.attention_dropout:
            A = Dropout(self.attention_dropout)(A)
        # 完成输出
        o = tf.einsum('bhjk,bkhd->bjhd', A, vw)
        if p_bias == 'typical_relative':
            o = o + tf.einsum('bhjk,jkd->bjhd', A, position_bias)
        return o, a

    def compute_output_shape(self, input_shape):
        o_shape = (input_shape[0][0], input_shape[0][1], self.out_dim)
        if self.return_attention_scores:
            a_shape = (
                input_shape[0][0], self.heads, input_shape[0][1],
                input_shape[1][1]
            )
            return [o_shape, a_shape]
        else:
            return o_shape

    def compute_mask(self, inputs, mask=None):
        if mask is not None:
            if self.return_attention_scores:
                return [mask[0], None]
            else:
                return mask[0]

    def get_config(self):
        config = {
            'heads': self.heads,
            'head_size': self.head_size,
            'out_dim': self.out_dim,
            'key_size': self.key_size,
            'use_bias': self.use_bias,
            'normalization': self.normalization,
            'attention_scale': self.attention_scale,
            'attention_dropout': self.attention_dropout,
            'return_attention_scores': self.return_attention_scores,
            'kernel_initializer':
                initializers.serialize(self.kernel_initializer),
        }
        base_config = super(MultiHeadAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

# 查看某一层的详细参数：
# In [27]: model.get_layer("Embedding-Token").__dict__

def build_bert_model( hidden_size= 312, intermediate_size= 1248,num_hidden_layers= 4, embedding_size=128):
    output_layer_num = num_hidden_layers
    Input_Token = Input(shape=(None,), name="Input-Token")
    Input_Segment = Input(shape=(None,), name="Input-Segment")

    Embedding_Token = Embedding(input_dim=13685, output_dim=embedding_size, mask_zero=True, name="Embedding-Token")(Input_Token)
    Embedding_Segment = Embedding(input_dim=2, output_dim=embedding_size, name="Embedding-Segment")(Input_Segment)

    Embedding_Token_Segment = Add(name="Embedding-Token-Segment")([Embedding_Token, Embedding_Segment])
    Embedding_Position = PositionEmbedding(input_dim=512, output_dim=embedding_size, mode='add',
                                           embeddings_regularizer=None,
                                           embeddings_constraint=None,
                                           mask_zero=False,
                                           name="Embedding-Position")(Embedding_Token_Segment)
    # Embedding_Dropout = Dropout(rate=0.1, name="Embedding-Dropout")(Embedding_Position)
    _Norm = LayerNormalization(name="Embedding-Norm", scale=True, offset= True, conditional= False, hidden_units= None,
                               hidden_activation= 'linear',hidden_initializer= 'TruncatedNormal',zero_mean= True,unit_variance= True,epsilon=1e-12,)(Embedding_Position)
    if num_hidden_layers == 12:
        linear = Dropout(rate=0.1, name="Embedding-Dropout")(_Norm)
    else:
        linear = Dense(hidden_size, activation='linear', use_bias=True, kernel_initializer='TruncatedNormal', bias_initializer='zeros', name="Embedding-Mapping",)(_Norm)
    for layer_index in range(0, output_layer_num ):
        multi_head_att = MultiHeadAttention(12, hidden_size//12, out_dim=hidden_size, key_size=hidden_size//12,
                                                    use_bias=True,
                                                    normalization='softmax',
                                                    attention_scale=True,
                                                    attention_dropout=0,
                                                    return_attention_scores=False,
                                                    kernel_initializer='TruncatedNormal',
                                                    name="Transformer-{}-MultiHeadSelfAttention".format(layer_index))([linear, linear, linear])
        multi_head_att_add = Add(name="Transformer-{}-MultiHeadSelfAttention-Add".format(layer_index))([multi_head_att, linear])
        multi_head_att_Norm = LayerNormalization(name="Transformer-{}-MultiHeadSelfAttention-Norm".format(layer_index), scale=True, offset=True, conditional=False, hidden_units=None,
                                   hidden_activation='linear', hidden_initializer='TruncatedNormal', zero_mean=True,
                                   unit_variance=True, epsilon=1e-12, )(multi_head_att_add)
        multi_head_att_feed = FeedForward(
                                        intermediate_size,name="Transformer-{}-FeedForward".format(layer_index),
                                        activation=gelu_erf,
                                        use_bias=True,
                                        kernel_initializer='TruncatedNormal',)(multi_head_att_Norm)
        multi_head_att_feed_add = Add(name="Transformer-{}-FeedForward-Add".format(layer_index))([multi_head_att_feed, multi_head_att_Norm])

        linear = LayerNormalization(name="Transformer-{}-FeedForward-Norm".format(layer_index), scale=True, offset= True, conditional= False, hidden_units= None,
                               hidden_activation= 'linear',hidden_initializer= 'TruncatedNormal',zero_mean= True,unit_variance= True,epsilon=1e-12,)(multi_head_att_feed_add)

    inputs = [Input_Token, Input_Segment]
    output = Lambda(lambda x: x[:, 0])(linear)
    model = Model(inputs, output)
    model.summary()
    return model

keras.backend.clear_session()
model = build_bert_model(hidden_size= 312, intermediate_size= 1248,num_hidden_layers= 4, embedding_size=128)
USERNAME = os.getenv('USERNAME')

def ckpt_to_hdf5():
    '''
    模型下载于：
    SimBERT-tiny
    https://github.com/ZhuiyiTechnology/pretrained-models
    https://open.zhuiyi.ai/releases/nlp/models/zhuiyi/chinese_simbert_L-4_H-312_A-12.zip
    转换为hdf5文件；
    环境要求：
    tensorflow 1.14 + keras 2.3.1 + bert4keras 0.7.7
    :return:
    '''
    import os
    import numpy as np
    from tqdm import tqdm
    from bert4keras.backend import keras, K
    from bert4keras.tokenizers import Tokenizer
    from bert4keras.models import build_transformer_model
    from bert4keras.optimizers import Adam, extend_with_piecewise_linear_lr
    # from bert4keras.utils import DataGenerator, pad_sequences
    from bert4keras.layers import *
    from bert4keras.models import *

    USERNAME = os.getenv("USERNAME")
    config_path = './result/pre-training/chinese_simbert_L-4_H-312_A-12/bert_config.json'
    checkpoint_path = './result/pre-training/chinese_simbert_L-4_H-312_A-12/bert_model.ckpt'
    dict_path = './result/pre-training/chinese_simbert_L-4_H-312_A-12/vocab.txt'

    # 建立分词器
    tokenizer = Tokenizer(dict_path, do_lower_case=True)

    cut_words = tokenizer.tokenize(u'今天天气不错')
    print(cut_words)

    simbert_model = build_transformer_model(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
    )

    output = Lambda(lambda x: x[:, 0])(simbert_model.output)
    model = Model(simbert_model.inputs, output)

    model.save(rf"D:\Users\{USERNAME}\data\chinese_simbert_L-4_H-312_A-12\SimBERT-tiny.hdf5")

# 加载环境（ tf.__version__ '1.14.0' keras.__version__ '2.3.1'）导出的hdf5文件权重；
model.load_weights(rf"D:\Users\{USERNAME}\data\chinese_simbert_L-4_H-312_A-12\SimBERT-tiny.hdf5")
# model2 = Model(model.inputs, model.get_layer('Transformer-0-MultiHeadSelfAttention').output)
a_token_ids = np.array([[2, 4342, 1775, 1760, 669, 6148, 4342, 5115, 675, 509, 3,
                         0, 0, 0, 0, 0],
                        [2, 695, 1370, 5437, 7914, 1775, 1760, 669, 3416, 7268, 7268,
                         4636, 3407, 3361, 675, 3]], dtype='int16')

a_vecs = model.predict([a_token_ids,
                        np.zeros_like(a_token_ids)],
                       verbose=True)
cosine_similarity(a_vecs)

# array([[0.9999999 , 0.86472136],
#        [0.86472136, 0.9999997 ]], dtype=float32)
model.save(rf"D:\Users\{USERNAME}\data\chinese_simbert_L-4_H-312_A-12\SimBERT-tiny_tf2.hdf5")

from keras.models import load_model
model2 = load_model(rf"D:\Users\{USERNAME}\data\chinese_simbert_L-4_H-312_A-12\SimBERT-tiny_tf2.hdf5",
                    custom_objects={"PositionEmbedding": PositionEmbedding, "FeedForward": FeedForward,
                                    "LayerNormalization": LayerNormalization, "MultiHeadAttention": MultiHeadAttention,
                                    "gelu_erf": gelu_erf,
                                    })

def main():
    pass


if __name__ == '__main__':
    main()
