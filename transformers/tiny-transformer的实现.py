#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 来源： https://keras.io/keras_nlp/

import keras_nlp
import tensorflow as tf
from tensorflow import keras

# Tokenize some inputs with a binary label.
vocab = ["[UNK]", "好", "学", "习", "天", "向", "上", "。"]
sentences = ['好 好 学 习 。', '天 天 向 上 。']
tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(
    vocabulary=vocab,
    sequence_length=10,
)
x, y = tokenizer(sentences), tf.constant([1, 0])

# Create a tiny transformer.
inputs = keras.Input(shape=(None,), dtype="int32")
outputs = keras_nlp.layers.TokenAndPositionEmbedding(
    vocabulary_size=len(vocab),
    sequence_length=10,
    embedding_dim=16,
)(inputs)
outputs = keras_nlp.layers.TransformerEncoder(
    num_heads=4,
    intermediate_dim=32,
)(outputs)
outputs = keras.layers.GlobalAveragePooling1D()(outputs)
outputs = keras.layers.Dense(1, activation="sigmoid")(outputs)
model = keras.Model(inputs, outputs)

# Run a single batch of gradient descent.
model.compile(optimizer="rmsprop", loss="binary_crossentropy", jit_compile=True)
model.train_on_batch(x, y)

def main():
    pass


if __name__ == '__main__':
    main()
