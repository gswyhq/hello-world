#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# https://keras.io/examples/timeseries/timeseries_classification_transformer/

# 加载数据集
# 我们将使用与从头开始的时间序列分类 示例相同的数据集和预处理 。
import os
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


def readucr(filename):
    data = np.loadtxt(filename, delimiter="\t")
    y = data[:, 0]
    x = data[:, 1:]
    return x, y.astype(int)


# root_url = "https://raw.githubusercontent.com/hfawaz/cd-diagram/master/FordA/"
root_url = r"D:\Users\{}\github_project\cd-diagram\FordA/".format(os.getenv("USERNAME"))
# 数据集来源： https://github.com/hfawaz/cd-diagram
x_train, y_train = readucr(root_url + "FordA_TRAIN.tsv")
x_test, y_test = readucr(root_url + "FordA_TEST.tsv")

x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

n_classes = len(np.unique(y_train))

idx = np.random.permutation(len(x_train))  # 打乱顺序
x_train = x_train[idx]
y_train = y_train[idx]

y_train[y_train == -1] = 0
y_test[y_test == -1] = 0


# 建立模型
# 我们的模型处理一个形状张量(batch size, sequence length, features)，其中sequence length是时间步数，features是每个输入时间序列。
#
# 你可以用这个替换你的分类 RNN 层：输入是完全兼容的！


# 我们包括残差连接、层归一化和 dropout。生成的层可以堆叠多次。

# 投影层通过 实现keras.layers.Conv1D。

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Attention and Normalization
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs, inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    return x + res


# 我们模型的主要部分现已完成。
# 我们可以堆叠多个这些 transformer_encoder块，还可以继续添加最终的多层感知器分类头。
# 除了一堆Dense 层，我们需要将TransformerEncoder模型部分的输出张量减少到当前批次中每个数据点的特征向量。
# 实现此目的的常用方法是使用池化层。
# 对于这个例子，GlobalAveragePooling1D一层就足够了。

def build_model(
    input_shape,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
    dropout=0,
    mlp_dropout=0,
):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs)


# 训练和评估
input_shape = x_train.shape[1:]

model = build_model(
    input_shape,
    head_size=256,
    num_heads=4,
    ff_dim=4,
    num_transformer_blocks=4,
    mlp_units=[128],
    mlp_dropout=0.4,
    dropout=0.25,
)

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    metrics=["sparse_categorical_accuracy"],
)
model.summary()

callbacks = [keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]

model.fit(
    x_train,
    y_train,
    validation_split=0.2,
    epochs=200,
    batch_size=64,
    callbacks=callbacks,
)

model.evaluate(x_test, y_test, verbose=1)

# 结论
# 在大约 110-120 个 epoch（Colab 上每个 25 秒）内，该模型达到了约 0.95 的训练准确度、约 84 的验证准确度和约 85 的测试准确度，无需进行超参数调整。
# 这适用于参数少于 100k 的模型。当然，参数计数和准确性可以通过超参数搜索和更复杂的学习率计划或不同的优化器来提高。

def main():
    pass


if __name__ == '__main__':
    main()
