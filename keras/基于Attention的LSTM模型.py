#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import keras
from keras import backend as K
from keras.layers import Input, Permute, Reshape, Dense, Lambda, Multiply, LSTM, Flatten, RepeatVector
from keras.models import Model

def attention_3d_block(inputs, SINGLE_ATTENTION_VECTOR=True):
    # inputs.shape = (batch_size, time_steps, input_dim)
    TIME_STEPS, input_dim = inputs.shape[-2:]
    a = Permute((2, 1))(inputs)  # 维度置换，将数据转化为(batch_size, input_dim, time_steps)
    a = Reshape((input_dim, TIME_STEPS))(a)
    a = Dense(TIME_STEPS, activation='softmax')(a)

    # 多维特征共享一个注意力权重，还是每一维特征单独有一个注意力权重
    if SINGLE_ATTENTION_VECTOR:
        # Lambda层将原本多维的注意力权重取平均，RepeatVector层再按特征维度复制粘贴，那么每一维特征的权重都是一样的了，也就是所说的共享一个注意力
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
        a = RepeatVector(input_dim)(a)  # RepeatVector层：作用为将输入重复n次
    a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = Multiply()([inputs, a_probs])
    return output_attention_mul


def model_attention_applied_before_lstm(TIME_STEPS, INPUT_DIM, SINGLE_ATTENTION_VECTOR=True):
    '''LSTM之前使用Attention'''
    K.clear_session()  # 清除之前的模型，省得压满内存
    inputs = Input(shape=(TIME_STEPS, INPUT_DIM,))
    attention_mul = attention_3d_block(inputs, SINGLE_ATTENTION_VECTOR)
    lstm_units = 32
    attention_mul = LSTM(lstm_units, return_sequences=False)(attention_mul)
    output = Dense(1, activation='sigmoid')(attention_mul)
    model = Model(inputs=[inputs], outputs=output)
    return model


def model_attention_applied_after_lstm(TIME_STEPS, INPUT_DIM, SINGLE_ATTENTION_VECTOR=True):
    '''LSTM之后使用Attention'''
    K.clear_session()  # 清除之前的模型，省得压满内存
    inputs = Input(shape=(TIME_STEPS, INPUT_DIM,))
    lstm_units = 32
    lstm_out = LSTM(lstm_units, return_sequences=True)(inputs)
    # 设置return_sequences=True, 注意此时LSTM的结构就不是N对1而是N对N了，
    # 因为要用Attention，所以输入到Attention里的特征要是多个才有意义。
    attention_mul = attention_3d_block(lstm_out, SINGLE_ATTENTION_VECTOR)
    attention_mul = Flatten()(attention_mul)
    output = Dense(1, activation='sigmoid')(attention_mul)
    model = Model(inputs=[inputs], outputs=output)
    return model


def attention_4d_block(inputs, SINGLE_ATTENTION_VECTOR=True, att_axis=-1):
    '''

    :param inputs:
    :param SINGLE_ATTENTION_VECTOR:
    :param att_axis: 添加注意力维度
    :return:
    '''
    # inputs.shape = (batch_size, time_steps, input_dim, vec_emd)
    time_steps, input_dim, vec_emd = inputs.shape[-3:]
    if att_axis in {-1, 3}:
        dims = (1, 2, 3)
    elif att_axis in {-2, 2}:
        dims = (1, 3, 2)
    elif att_axis in {-3, 1}:
        dims = (3, 2, 1)
    else:
        raise ValueError('不支持的维度：{}'.format(att_axis))
    a = Permute(dims)(inputs)  # 维度置换
    #     print(a.shape)
    a = Dense(a.shape[-1], activation='softmax')(a)

    #     # 多维特征共享一个注意力权重，还是每一维特征单独有一个注意力权重
    #     if SINGLE_ATTENTION_VECTOR:
    #         # Lambda层将原本多维的注意力权重取平均，RepeatVector层再按特征维度复制粘贴，那么每一维特征的权重都是一样的了，也就是所说的共享一个注意力
    #         a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
    #         a = RepeatVector(input_dim)(a)  # RepeatVector层：作用为将输入重复n次
    a_probs = Permute(dims)(a)
    output_attention_mul = Multiply()([inputs, a_probs])
    return output_attention_mul


K.clear_session()  # 清除之前的模型，省得压满内存
inputs = Input(shape=(6, 74, 32))
attention_mul = attention_4d_block(inputs, att_axis=-2)
print(attention_mul)

def main():
    TIME_STEPS, INPUT_DIM = 6, 74
    model = model_attention_applied_before_lstm(TIME_STEPS, INPUT_DIM, SINGLE_ATTENTION_VECTOR=True)
    model.summary()

    model = model_attention_applied_after_lstm(TIME_STEPS, INPUT_DIM, SINGLE_ATTENTION_VECTOR=True)
    model.summary()

    model = model_attention_applied_before_lstm(TIME_STEPS, INPUT_DIM, SINGLE_ATTENTION_VECTOR=False)
    model.summary()

    model = model_attention_applied_after_lstm(TIME_STEPS, INPUT_DIM, SINGLE_ATTENTION_VECTOR=False)
    model.summary()

if __name__ == '__main__':
    main()
