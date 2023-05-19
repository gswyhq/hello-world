#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import numpy as np
import math
import pandas as pd
import keras
from keras.layers import Input, Lambda, Dense
from keras.models import Model, load_model
import keras.backend as K
import tensorflow as tf
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
from tensorflow.keras.utils import Sequence
from keras.callbacks import History
from tqdm import tqdm
from keras.optimizers import adam_v2
from keras.models import load_model
from sklearn.metrics.pairwise import cosine_similarity

USERNAME = os.getenv('USERNAME')

config_path = r'D:\Users\{}\data\RoBERTa-tiny-clue\bert_config.json'.format(os.getenv("USERNAME"))
checkpoint_path = r'D:\Users\{}\data\RoBERTa-tiny-clue\bert_model.ckpt'.format(os.getenv("USERNAME"))
dict_path = r'D:\Users\{}\data\RoBERTa-tiny-clue\vocab.txt'.format(os.getenv("USERNAME"))

def cosent_loss(y_true, y_pred):
    """排序交叉熵
    y_true：标签/打分，y_pred：句向量
    """
    y_true = y_true[::2, 0]  # 获取偶数位标签，即取出真实的标签；
    y_true = K.cast(y_true[:, None] < y_true[None, :], K.floatx())  # 取出负例-正例的差值
    y_pred = K.l2_normalize(y_pred, axis=1)  # 对输出的句子向量进行l2归一化   后面只需要对应位相乘  就可以得到cos值了
    y_pred = K.sum(y_pred[::2] * y_pred[1::2], axis=1) * 20  # 奇偶位向量相乘，得到对应cos
    y_pred = y_pred[:, None] - y_pred[None, :]  # 取出负例-正例的差值, # 这里是算出所有位置 两两之间余弦的差值
    # 矩阵中的第i行j列  表示的是第i个余弦值-第j个余弦值
    y_pred = K.reshape(y_pred - (1 - y_true) * 1e12, [-1])  # 乘以e的12次方,要排除掉不需要计算(mask)的部分
    y_pred = K.concatenate([[0], y_pred], axis=0)  # 这里加0是因为e^0 = 1相当于在log中加了1
    return K.logsumexp(y_pred)

save_h5_file = rf"D:\Users\{USERNAME}\data\客户标签相似性\pb或pt模型\20230210181050\text_similarity_11-0.6371.hdf5"
save_tflite_file = rf"D:\Users\{USERNAME}\data\客户标签相似性\pb或pt模型\20230210181050\text_similarity_11-0.6371.tflite"
# 构建模型
# base = build_transformer_model(config_path, checkpoint_path)
bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=None)
###########################################################################################################################
custom_objects = {layer.__class__.__name__:layer for layer in bert_model.layers}
custom_objects['cosent_loss'] =cosent_loss
encoder2 = load_model(save_h5_file, custom_objects=custom_objects)
encoder2.inputs[0].set_shape((None, 16))
encoder2.inputs[1].set_shape((None, 16))

test_data = {
    "Input-Token": [
        [101, 2741, 1222, 3799, 2222, 242, 254, 6556, 102, 0, 0, 0, 0, 0, 0, 0],
        [101, 2741, 1222, 1676, 2222, 242, 254, 6556, 102, 0, 0, 0, 0, 0, 0, 0]],
    "Input-Segment": [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
             }


input_list = [np.reshape(np.array(test_data["Input-Segment"], dtype='float32')[0,:], (1, 16))
, np.reshape(np.array(test_data["Input-Token"], dtype='float32')[0,:], (1, 16))]
encoder2._set_inputs(input_list)

a_vecs = encoder2.predict([np.array(test_data["Input-Token"]), np.array(test_data["Input-Segment"])])
cosine_similarity(a_vecs)
# array([[1.0000001, 0.9938997],
#        [0.9938997, 1.0000005]]

from tensorflow import keras
from tensorflow import lite

# converter = lite.TocoConverter.from_keras_model_file(keras_file)
converter = lite.TFLiteConverter.from_keras_model(encoder2)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS
]
converter.target_spec.supported_types = [tf.float32] # 有时候结果不对，可能需要指定数据类型，否则默认是int 类型

converter.inference_input_type = tf.float32
converter.inference_output_type = tf.float32
converter.post_training_quantize = True
tflite_model = converter.convert()
# Function `_wrapped_model` contains input name(s) Input-Token, Input-Segment with unsupported characters which will be renamed to input_token, input_segment in the SavedModel.
open(save_tflite_file, "wb").write(tflite_model)


# 加载tflite模型进行预测：
# Load the TFLite model in TFLite Interpreter
print(os.path.isfile(save_tflite_file))
interpreter = tf.lite.Interpreter(model_path=save_tflite_file)  # 有时候加载模型不成功，可能是因为tensorflow版本不一致，或者是因为不是linux系统所致；
interpreter.allocate_tensors()
inputs = interpreter.get_input_details()
outputs = interpreter.get_output_details()

interpreter.set_tensor(inputs[0]['index'], np.reshape(np.array(test_data["Input-Token"], dtype='float32')[0,:], (1, 16)))
interpreter.set_tensor(inputs[1]['index'], np.reshape(np.array(test_data["Input-Segment"], dtype='float32')[0,:], (1, 16)))
interpreter.invoke()
b_vecs = interpreter.get_tensor(outputs[0]['index'])
b_vecs.sum()
# Out[15]: -9.328365

def main():
    pass


if __name__ == '__main__':
    main()
