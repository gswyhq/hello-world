#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import keras
from tensorflow.python.keras.utils import np_utils
import tensorflow as tf
from keras import backend as K
from keras.engine.input_layer import InputLayer
from keras.layers import Input, Lambda, Reshape, Flatten, Dense, Concatenate, Dropout
from keras.models import Model
from keras_bert.layers.embedding import TokenEmbedding
# from keras.layers.embeddings import Embedding
# from keras.layers.merge import Add
from keras.layers.core.embedding import Embedding
from keras.layers.merging.add import Add
from keras_pos_embd.pos_embd import PositionEmbedding
from keras.layers.core import Dropout
from keras_layer_normalization.layer_normalization import LayerNormalization
from keras_multi_head.multi_head_attention import MultiHeadAttention
from keras_position_wise_feed_forward.feed_forward import FeedForward
from keras import initializers


import pydot
from keras.utils.vis_utils import plot_model
from IPython.display import Image
from keras.models import Model

'''
图像 (ViT ) 和音频分类 (AST) 层次结构
ViT和AST采用了 Transformer 结构，能够处理序列特征。首先从RGB图像 (或音频频谱图) 中提取N个不重叠的patch（patch可以通俗地理解为图像块），
然后将他们转换成一维的token。z = [CLS, 每个token映射, 特征的位置信息]
首位[CLS]是一个特殊的token，作为分类任务的特征；
一维的token还包括用于表示输入特征的位置信息,是可学习的位置嵌入。
然后将token通过由L个 Transformer 层组成的编码器中。
每个Transformer层由多头自注意 (MSA)，层归一化 (LN) 和多层感知机 (MLP) 组成。
y = MSA(LN(z)) + z
Z(l+1) = MLP(LN(y)) + y

普通的融合模型仅由应用于多模态输入的常规Transformer组成。
对于给定长度为t秒的视频clip，首先统一采样F个RGB帧，并将音频波形转换为单个谱图。然后用类似ViT中的方法，将帧和谱图转换成token，并将所有的token拼接在一起，成为一个序列。
形式上，如果从F个采样帧里面提出了Nv 个RGB patch和Na个谱图patch。则输入的token序列可以表示为z=[Zrgb‖Zspec]，
对于RGB patch和谱图patch，作者采用了不同的投影函数Ergb,Espec。此外，还为每个模态分配了一个分类token。然后在这些多模态token上采用Transformer层，以获取跨模态的融合信息
为了克服attention的平方复杂度，作者在输入序列中引入了个瓶颈token, 输入序列如下所示 z = [Zrgb ‖ Zfsn ‖ Zspec] 。
然后，用这些瓶颈token来限制模型中的所有跨模态注意力。对于第l层的Transformer，token计算如下：
其中，z{rgb}, z{spec}通过Transformer层内的瓶颈token z{fsn}交换信息。由于B≪Nv,B≪Na，因此融合过程的计算量可以大大降低。

首先用Lf个标准的Self-Attention层来对模态内的token信息进行建模，然后再将所有的token进行拼接得到z=[zrgb||zspec]，用剩下的L−Lf层进行跨模态token信息的融合。
如果Lf=0，那么就对应“早期融合”；如果Lf=L，那么就对应“晚期融合”；如果 0<Lf<L,那么就对应“中期融合”。

为了能够执行分类任务，需要将最后一层的CLS token zcls-rgb和zcls-spec输入到线性层，然后将Softmax之后的结果进行平均得到分类结果。

'''

def transformer_encoder(_Norm, layer_index, name=''):
    MultiHeadSelfAttention = MultiHeadAttention(head_num=4,
                                                name="{}-Encoder-{}-MultiHeadSelfAttention".format(name, layer_index))(_Norm)
    MultiHeadSelfAttention_Dropout = Dropout(rate=0.1,
                                             name="{}-Encoder-{}-MultiHeadSelfAttention-Dropout".format(name, layer_index))(
        MultiHeadSelfAttention)
    MultiHeadSelfAttention_Add = Add(name="{}-Encoder-{}-MultiHeadSelfAttention-Add".format(name, layer_index))(
        [_Norm, MultiHeadSelfAttention_Dropout])
    MultiHeadSelfAttention_Norm = LayerNormalization(
        name="{}-Encoder-{}-MultiHeadSelfAttention-Norm".format(name, layer_index))(MultiHeadSelfAttention_Add)
    feedForward = FeedForward(units=64, name="{}-Encoder-{}-FeedForward".format(name, layer_index))(
        MultiHeadSelfAttention_Norm)
    FeedForward_Dropout = Dropout(rate=0.1, name="{}-Encoder-{}-FeedForward-Dropout".format(name, layer_index))(feedForward)
    FeedForward_Add = Add(name="{}-Encoder-{}-FeedForward-Add".format(name, layer_index))(
        [MultiHeadSelfAttention_Norm, FeedForward_Dropout])
    _Norm = LayerNormalization(name="{}-Encoder-{}-FeedForward-Norm".format(name, layer_index))(FeedForward_Add)
    return _Norm

def encoder(_Norm, output_layer_num, name=''):
    for layer_index in range(1, output_layer_num + 1):
        _Norm = transformer_encoder(_Norm, layer_index, name=name)
    return _Norm

def fusion_layer(rgb, spec, fsn, fusion_layer_num=4):

    for layer_index in range(1, fusion_layer_num + 1):
        rgb_fsn = Concatenate()([rgb, fsn])

        rgb_fsn_norm = transformer_encoder(rgb_fsn, layer_index, name="RGB-FSN")
        # rgb, fsn = np.split(rgb_fsn_norm, [rgb.shape[-1], ], axis=2)
        # rgb, fsn = tf.split(value=rgb_fsn_norm, num_or_size_splits=rgb.shape[-1], axis=-1)
        rgb, fsn = Lambda(tf.split, arguments={'axis': -1, 'num_or_size_splits': [rgb.shape[-1], fsn.shape[-1]]})(rgb_fsn_norm)
        spec_fsn = Concatenate()([spec, fsn])
        spec_fsn_norm = transformer_encoder(spec_fsn, layer_index, name="SPEC-FSN")
        # spec, fsn = tf.split(value=spec_fsn_norm, num_or_size_splits=spec.shape[-1], axis=-1)
        spec, fsn = Lambda(tf.split, arguments={'axis': -1, 'num_or_size_splits': [spec.shape[-1], fsn.shape[-1]]})(spec_fsn_norm)
        # output = Lambda(lambda x: K.mean(x, axis=1), name="lambda")(_Norm)

    return rgb, spec, fsn

def model1(name='', output_dim=312, input_dim=1008):
    Input_Token = Input(shape=(None,), name="Input-{}-Token".format(name))
    Input_Segment = Input(shape=(None,), name="Input-{}-Segment".format(name))

    Embedding_Token = TokenEmbedding(input_dim=input_dim, output_dim=output_dim, name="{}-Embedding-Token".format(name))(Input_Token)
    Embedding_Segment = Embedding(input_dim=2, output_dim=output_dim, name="{}-Embedding-Segment".format(name))(Input_Segment)

    Embedding_Token_Segment = Add(name="{}-Embedding-Token-Segment".format(name))([Embedding_Token[0], Embedding_Segment])
    Embedding_Position = PositionEmbedding(input_dim=128, output_dim=output_dim, mode='add',
                                           embeddings_regularizer=None,
                                           embeddings_constraint=None,
                                           mask_zero=False,
                                           name="{}-Embedding-Position".format(name))(Embedding_Token_Segment)
    Embedding_Dropout = Dropout(rate=0.1, name="{}-Embedding-Dropout".format(name))(Embedding_Position)
    _Norm = LayerNormalization(name="{}-Embedding-Norm".format(name))(Embedding_Dropout)
    return Input_Token, Input_Segment, _Norm

def build_mbt_model(output_layer_num=4, metrics=None, output_dim=312, fsn_units=32, fusion_layer_num=4, rgb_dim=255, spec_dim=300):
    Input_FSN = Input(shape=(None, 1), name="Input-FSN")
    Input_RGB_Token, Input_RGB_Segment, rgb_norm = model1(name="RGB", output_dim=output_dim, input_dim=rgb_dim)
    Input_SPEC_Token, Input_SPEC_Segment, spec_norm = model1(name="SPEC", output_dim=output_dim, input_dim=spec_dim)
    rgb = encoder(rgb_norm, output_layer_num, name='RGB')
    spec = encoder(spec_norm, output_layer_num, name='SPEC')
    print("_Norm", rgb, spec)
    fsn = Dense(fsn_units, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=0.02))(Input_FSN)
    # output = Lambda(lambda x: K.mean(x, axis=1), name="lambda")(_Norm)

    rgb, spec, fsn = fusion_layer(rgb, spec, fsn, fusion_layer_num)
    rgb_cls = Lambda(lambda x: x[:, 0])(rgb)
    spec_cls = Lambda(lambda x: x[:, 0])(spec)
    fusion_cls = Concatenate()([rgb_cls, spec_cls])
    x = Dropout(0.1)(fusion_cls)
    x = Dense(16, activation="relu")(x)
    x = Dropout(0.1)(x)
    output = Dense(2, activation="softmax")(x)
    inputs = [Input_RGB_Token, Input_RGB_Segment, Input_FSN, Input_SPEC_Token, Input_SPEC_Segment]
    outputs = [output]
    model = Model(inputs, outputs)
    #     adam = keras.optimizers.Adam(decay=0.2)
    if metrics is None:
        metrics = ['accuracy']
    model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=metrics) # categorical_crossentropy, binary_crossentropy
    model.summary()
    return model

keras.backend.clear_session()
rgb_dim=255
spec_dim=300
mbt_model = build_mbt_model(output_layer_num=4, metrics=None, output_dim=64, rgb_dim=rgb_dim, spec_dim=spec_dim)

# plot_model(mbt_model, show_shapes=True, show_layer_names=True, to_file=r"1233321.png")

train_size = 1000
RGB_Token = np.random.randint(rgb_dim, size=(train_size, 64))
SPEC_Token = np.random.randint(spec_dim, size=(train_size, 64))
train_x = {'Input-RGB-Token': RGB_Token,
 'Input-RGB-Segment': np.zeros_like(RGB_Token),
 'Input-FSN': np.ones((train_size, 64)),
 'Input-SPEC-Token': SPEC_Token,
 'Input-SPEC-Segment': np.zeros_like(SPEC_Token)}

train_y = np.reshape([np.random.randint(0, 2) for _ in range(train_size)], (train_size, 1))
train_y = np_utils.to_categorical(train_y, 2)

mbt_model.fit(train_x, train_y, batch_size=16, validation_split= 0.1, epochs=1, verbose=1,)


def main():
    pass


if __name__ == '__main__':
    main()
