#!/usr/bin/env python


import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io
from pathlib import Path
from absl import app
from absl import flags
from absl import logging
import keras
from keras import backend as K
from keras.models import model_from_json, model_from_yaml


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 将keras、tensorflow2.x训练的h5文件转换为pb文件
# 若是 tensorflow1.x,直接使用 https://github.com/amir-abdi/keras_to_tensorflow 脚本即可
# tf.__version__
# Out[27]: '2.6.0'
# keras.__version__
# Out[28]: '2.6.0'

class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_dim=32, num_heads=8,
                 projection_dim= 4,
                    query_dense = layers.Dense(32),
                    key_dense = layers.Dense(32),
                    value_dense = layers.Dense(32),
                    combine_heads = layers.Dense(32),
                 **kwargs):
        super(MultiHeadSelfAttention, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads
        self.query_dense = layers.Dense(embed_dim)
        self.key_dense = layers.Dense(embed_dim)
        self.value_dense = layers.Dense(embed_dim)
        self.combine_heads = layers.Dense(embed_dim)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'embed_dim' : self.embed_dim,
            'num_heads' : self.num_heads,
            'projection_dim' : self.projection_dim,
            'query_dense' : self.query_dense,
            'key_dense' : self.key_dense,
            'value_dense' : self.value_dense,
            'combine_heads' : self.combine_heads,
        })
        return config

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        # x.shape = [batch_size, seq_len, embedding_dim]
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)  # (batch_size, seq_len, embed_dim)
        key = self.key_dense(inputs)  # (batch_size, seq_len, embed_dim)
        value = self.value_dense(inputs)  # (batch_size, seq_len, embed_dim)
        query = self.separate_heads(
            query, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        key = self.separate_heads(
            key, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        value = self.separate_heads(
            value, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(
            attention, perm=[0, 2, 1, 3]
        )  # (batch_size, seq_len, num_heads, projection_dim)
        concat_attention = tf.reshape(
            attention, (batch_size, -1, self.embed_dim)
        )  # (batch_size, seq_len, embed_dim)
        output = self.combine_heads(
            concat_attention
        )  # (batch_size, seq_len, embed_dim)
        return output


'''Transformer的Encoder部分'''


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim=32, num_heads=2, ff_dim=32, rate=0.1,
                 att = MultiHeadSelfAttention(32, 2),
                ffn = keras.Sequential(
                    [layers.Dense(32, activation="relu"), layers.Dense(32), ]
                ),
                layernorm1 = layers.LayerNormalization(epsilon=1e-6),
                layernorm2 = layers.LayerNormalization(epsilon=1e-6),
                dropout1 = layers.Dropout(0.1),
                dropout2 = layers.Dropout(0.1),
                 **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim), ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
                    "att": self.att,
                    "ffn": self.ffn ,
                    "layernorm1": self.layernorm1,
                    "layernorm2": self.layernorm2 ,
                    "dropout1": self.dropout1,
                    "dropout2": self.dropout2 ,
        })
        return config

'''Transformer输入的编码层'''


class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen=200, vocab_size=20000, embed_dim=32,
                 token_emb= layers.Embedding(input_dim=20000, output_dim=32),
                 pos_emb= layers.Embedding(input_dim=200, output_dim=32),
                 **kwargs):
        super(TokenAndPositionEmbedding, self).__init__(**kwargs)
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'token_emb': self.token_emb,
            'pos_emb': self.pos_emb,
        })
        return config

import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

def h5_to_pb(h5_save_path, pb_model_path="model.pb"):
    model = tf.keras.models.load_model(h5_save_path, compile=False,
                                       custom_objects={'TokenAndPositionEmbedding': TokenAndPositionEmbedding,
                                                       'TransformerBlock': TransformerBlock,
                                                       "MultiHeadSelfAttention": MultiHeadSelfAttention,
                                                       # "token_emb": layers.Embedding(input_dim=20000, output_dim=32),
                                                       # "pos_emb": layers.Embedding(input_dim=200, output_dim=32)
                                                       }  # custom_objects, 参数设置自定义网络层；
                                       )
    model.summary()
    # full_model = tf.function(lambda Input: model(Input))
    # full_model = full_model.get_concrete_function(tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))

    # 支持多输入模型：
    full_model = tf.function(lambda *x: model(x))
    x_tensor_spec = [tf.TensorSpec(shape=input_x.shape, dtype=input_x.dtype) for input_x in model.inputs]
    # 使用get_concrete_function函数把一个tf.function标注的普通的python函数变成带有图定义的函数。
    full_model = full_model.get_concrete_function(*x_tensor_spec)

    # Get frozen ConcreteFunction
    frozen_func = convert_variables_to_constants_v2(full_model)
    frozen_func.graph.as_graph_def()

    layers = [op.name for op in frozen_func.graph.get_operations()]
    print("-" * 50)
    print("Frozen model layers: ")
    for layer in layers:
        print(layer)

    print("-" * 50)
    print("Frozen model inputs: ")
    print(frozen_func.inputs)
    print("Frozen model outputs: ")
    print(frozen_func.outputs)

    # tensorflow 1.x 通过如下方法查看
    # print('input is :', [t.name for t in model.inputs])
    # print ('output is:', [t.name for t in model.outputs])

    # Save frozen graph from frozen ConcreteFunction to hard drive
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                      logdir=".",
                      name=pb_model_path,
                      as_text=False)   #可设置.pb存储路径

def main():
    h5_to_pb(r"imdb.h5", pb_model_path=r"imdb.pb")

if __name__ == "__main__":
    main()
