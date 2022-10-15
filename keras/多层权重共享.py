#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 实现权重共享需要将图层实例化与模型创建分离。
# 首先创建每个层，然后使用函数式API遍历它们来创建模型。

from tensorflow.keras import layers, initializers, Model

def embedding_block(dim):
    dense = layers.Dense(dim, activation=None, kernel_initializer='glorot_normal')
    activ = layers.PReLU(alpha_initializer=initializers.Constant(value=0.25))
    bnorm = layers.BatchNormalization()
    return [dense, activ, bnorm]

stack = embedding_block(8) + embedding_block(16) + embedding_block(32)

inp1 = layers.Input((5,))
inp2 = layers.Input((5,))

x,y = inp1,inp2
for layer in stack:
    x = layer(x)
    y = layer(y)

concat_layer = layers.Concatenate()([x,y])
pred = layers.Dense(1, activation="sigmoid")(concat_layer)

model = Model(inputs = [inp1, inp2], outputs=pred)


####################################################复杂权重共享示例#############################################################################
from keras.engine.input_layer import InputLayer
from keras_bert.layers.embedding import TokenEmbedding
# from keras.layers.embeddings import Embedding  # keras version==2.6.0
# from keras.layers.merge import Add
from keras.layers.core.embedding import Embedding  # keras version == 2.9.0
from keras.layers.merging.add import Add
from keras_pos_embd.pos_embd import PositionEmbedding
from keras.layers.core import Dropout
from keras_layer_normalization.layer_normalization import LayerNormalization
from keras_multi_head.multi_head_attention import MultiHeadAttention
from keras_position_wise_feed_forward.feed_forward import FeedForward
from keras.layers import Dense, Dropout, Lambda, Input, Concatenate
import keras.backend as K

# 第一步图层实例化
def defined_shared_layers(output_layer_num=4, output_dim=64):
    shared_layers_dict = {}
    Embedding_Token = TokenEmbedding(input_dim=1008, output_dim=output_dim, name="Embedding-Token")
    Embedding_Segment = Embedding(input_dim=2, output_dim=output_dim, name="Embedding-Segment")
    Embedding_Token_Segment = Add(name="Embedding-Token-Segment")
    Embedding_Position = PositionEmbedding(input_dim=128, output_dim=output_dim, mode= 'add',
                                             embeddings_regularizer= None,
                                             embeddings_constraint= None,
                                             mask_zero= False,
                                           name="Embedding-Position")
    Embedding_Dropout = Dropout(rate=0.1, name="Embedding-Dropout")
    _Norm = LayerNormalization(name="Embedding-Norm")
    for shared_layer in [Embedding_Token, Embedding_Segment, Embedding_Token_Segment, Embedding_Position, Embedding_Dropout, _Norm]:
        shared_layers_dict[shared_layer.name] = shared_layer
    for layer_index in range(1, output_layer_num+1):
        MultiHeadSelfAttention = MultiHeadAttention(head_num=4, name="Encoder-{}-MultiHeadSelfAttention".format(layer_index))
        MultiHeadSelfAttention_Dropout = Dropout(rate=0.1, name="Encoder-{}-MultiHeadSelfAttention-Dropout".format(layer_index))
        MultiHeadSelfAttention_Add = Add(name="Encoder-{}-MultiHeadSelfAttention-Add".format(layer_index))
        MultiHeadSelfAttention_Norm = LayerNormalization(name="Encoder-{}-MultiHeadSelfAttention-Norm".format(layer_index))
        feedForward = FeedForward(units=64, name="Encoder-{}-FeedForward".format(layer_index))
        FeedForward_Dropout = Dropout(rate=0.1, name="Encoder-{}-FeedForward-Dropout".format(layer_index))
        FeedForward_Add = Add(name="Encoder-{}-FeedForward-Add".format(layer_index))
        _Norm = LayerNormalization(name="Encoder-{}-FeedForward-Norm".format(layer_index))
        for shared_layer in [MultiHeadSelfAttention, MultiHeadSelfAttention_Dropout, MultiHeadSelfAttention_Add, MultiHeadSelfAttention_Norm,
                              feedForward, FeedForward_Dropout, FeedForward_Add, _Norm]:
            shared_layers_dict[shared_layer.name] = shared_layer
    lamb = keras.layers.Lambda(lambda x: K.mean(x, axis=1), name="lambda" )
    drop = Dropout(0.2, name="dropout")
    dense1 = Dense(16, name="dense", kernel_initializer="uniform", activation='relu')
    for shared_layer in [lamb, drop, dense1]:
        shared_layers_dict[shared_layer.name] = shared_layer
    return shared_layers_dict
SHARED_LAYERS_DICT = defined_shared_layers(output_layer_num=4, output_dim=64)  # 共享层实例化

# 构建网络
def build_shared_network(Input_Token, Input_Segment, output_layer_num=4):
    Embedding_Token = SHARED_LAYERS_DICT["Embedding-Token"](Input_Token)
    Embedding_Segment = SHARED_LAYERS_DICT["Embedding-Segment"](Input_Segment)

    Embedding_Token_Segment = SHARED_LAYERS_DICT["Embedding-Token-Segment"]([Embedding_Token[0], Embedding_Segment])
    Embedding_Position = SHARED_LAYERS_DICT["Embedding-Position"](Embedding_Token_Segment)
    Embedding_Dropout = SHARED_LAYERS_DICT["Embedding-Dropout"](Embedding_Position)
    _Norm = SHARED_LAYERS_DICT["Embedding-Norm"](Embedding_Dropout)

    for layer_index in range(1, output_layer_num + 1):
        MultiHeadSelfAttention = SHARED_LAYERS_DICT["Encoder-{}-MultiHeadSelfAttention".format(layer_index)](_Norm)
        MultiHeadSelfAttention_Dropout = SHARED_LAYERS_DICT[
            "Encoder-{}-MultiHeadSelfAttention-Dropout".format(layer_index)](MultiHeadSelfAttention)
        MultiHeadSelfAttention_Add = SHARED_LAYERS_DICT["Encoder-{}-MultiHeadSelfAttention-Add".format(layer_index)](
            [_Norm, MultiHeadSelfAttention_Dropout])
        MultiHeadSelfAttention_Norm = SHARED_LAYERS_DICT["Encoder-{}-MultiHeadSelfAttention-Norm".format(layer_index)](
            MultiHeadSelfAttention_Add)
        feedForward = SHARED_LAYERS_DICT["Encoder-{}-FeedForward".format(layer_index)](MultiHeadSelfAttention_Norm)
        FeedForward_Dropout = SHARED_LAYERS_DICT["Encoder-{}-FeedForward-Dropout".format(layer_index)](feedForward)
        FeedForward_Add = SHARED_LAYERS_DICT["Encoder-{}-FeedForward-Add".format(layer_index)](
            [MultiHeadSelfAttention_Norm, FeedForward_Dropout])
        _Norm = SHARED_LAYERS_DICT["Encoder-{}-FeedForward-Norm".format(layer_index)](FeedForward_Add)

    lamb = SHARED_LAYERS_DICT["lambda"](_Norm)
    drop = SHARED_LAYERS_DICT["dropout"](lamb)
    dense1 = SHARED_LAYERS_DICT["dense"](drop)
    return dense1


def build_bert_model3(metrics=None):
    Input_Token_a = Input(shape=(None,), name="Input-Token-a")
    Input_Segment_a = Input(shape=(None,), name="Input-Segment-a")
    Input_Token_b = Input(shape=(None,), name="Input-Token-b")
    Input_Segment_b = Input(shape=(None,), name="Input-Segment-b")

    dense_a = build_shared_network(Input_Token_a, Input_Segment_a)
    dense_b = build_shared_network(Input_Token_b, Input_Segment_b)
    concat_layer = Concatenate()([dense_a, dense_b, K.abs(dense_a - dense_b)])

    output = Dense(1, kernel_initializer="uniform", activation='sigmoid', name="output")(concat_layer)
    inputs = [Input_Token_a, Input_Segment_a, Input_Token_b, Input_Segment_b]
    outputs = [output]
    model = Model(inputs, outputs)
    #     adam = keras.optimizers.Adam(decay=0.2)
    if metrics is None:
        metrics = ['accuracy']
    model.compile(loss='binary_crossentropy', optimizer='adam',
                  metrics=metrics)  # categorical_crossentropy, binary_crossentropy
    model.summary()
    return model


keras.backend.clear_session()
shared_model = build_bert_model3(metrics=None)

def main():
    pass


if __name__ == '__main__':
    main()
