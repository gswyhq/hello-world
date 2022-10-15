#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# https://keras.io/examples/structured_data/classification_with_grn_and_vsn/

# 介绍
# 此示例演示了 Bryan Lim 等人提出的门控残差网络 (GRN) 和变量选择网络 (VSN) 的使用。
# 在 用于可解释多水平时间序列预测的时间融合变压器 (TFT) 中，用于结构化数据分类。
# GRN 使模型能够灵活地仅在需要时应用非线性处理。VSN 允许模型轻柔地删除任何可能对性能产生负面影响的不必要的噪声输入。
# 总之，这些技术有助于提高深度神经网络模型的学习能力。
# 请注意，此示例仅实现了论文中描述的 GRN 和 VSN 组件，而不是整个 TFT (Temporal Fusion Transformers)模型，因为 GRN 和 VSN 可以单独用于结构化数据学习任务。

# 要运行代码，您需要使用 TensorFlow 2.3 或更高版本。
#
# 准备数据
# 数据来源：https://archive.ics.uci.edu/ml/machine-learning-databases/adult/
# 此示例使用 UC Irvine Machine Learning Repository 提供的 美国人口普查收入数据集。该任务是二元分类来预测一个人是否有可能年收入超过 50,000 美元。
#
# 该数据集包括 48,842 个实例，具有 14 个输入特征：5 个数字特征和 9 个分类特征。

# 设置
import os
import math
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 准备数据
# 首先，我们将 UCI 机器学习存储库中的数据加载到 Pandas DataFrame 中。

CSV_HEADER = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "gender",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income_bracket",
]
USERNAME = os.getenv("USERNAME")

# 数据来源：https://archive.ics.uci.edu/ml/machine-learning-databases/adult/

train_data_url = rf"D:\Users\{USERNAME}\data\adult\adult.data"
train_dev_data = pd.read_csv(train_data_url, header=None, names=CSV_HEADER)

test_data_url = rf"D:\Users\{USERNAME}\data\adult\adult.test"
test_data = pd.read_csv(test_data_url, header=None, names=CSV_HEADER)

print(f"Train dataset shape: {train_dev_data.shape}")
print(f"Test dataset shape: {test_data.shape}")

# 删除第一条记录（因为它不是有效的数据示例）和类标签中的尾随“点”。

test_data = test_data[1:]
test_data.income_bracket = test_data.income_bracket.apply(
    lambda value: value.replace(".", "")
)

# 目标特征（target feature）名称：
TARGET_FEATURE_NAME = "income_bracket"


# 我们将目标列从字符串转换为整数。

train_dev_data[TARGET_FEATURE_NAME] = train_dev_data[TARGET_FEATURE_NAME].apply(
    lambda x: 0 if x == ' <=50K' else 1
)
test_data[TARGET_FEATURE_NAME] = test_data[TARGET_FEATURE_NAME].apply(
    lambda x: 0 if x == ' <=50K' else 1
)

# 然后，我们将数据集拆分为训练集和验证集。

random_selection = np.random.rand(len(train_dev_data.index)) <= 0.85
train_data = train_dev_data[random_selection]
valid_data = train_dev_data[~random_selection]

# 最后，我们将训练和测试数据拆分本地存储到 CSV 文件中。
train_data_file = rf"D:\Users\{USERNAME}\data\adult\train_data.csv"
valid_data_file = rf"D:\Users\{USERNAME}\data\adult\valid_data.csv"
test_data_file = rf"D:\Users\{USERNAME}\data\adult\test_data.csv"

train_data.to_csv(train_data_file, index=False, header=False)
valid_data.to_csv(valid_data_file, index=False, header=False)
test_data.to_csv(test_data_file, index=False, header=False)

# 定义数据集元数据
# 在这里，我们定义了数据集的元数据，这些元数据对于读取数据并将其解析为输入特征以及根据输入特征的类型对输入特征进行编码很有用。


# 用作实例权重的列的名称。Name of the column to be used as instances weight.
WEIGHT_COLUMN_NAME = "fnlwgt"

# 数值特征(numerical feature)列表：
NUMERIC_FEATURE_NAMES = [
    "age",
    "education_num",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
]

# 类别特征（categorical features）及其可取值字典
CATEGORICAL_FEATURES_WITH_VOCABULARY = {
    "workclass": sorted(list(train_dev_data["workclass"].unique())),
    "education": sorted(list(train_dev_data["education"].unique())),
    "marital_status": sorted(list(train_dev_data["marital_status"].unique())),
    "occupation": sorted(list(train_dev_data["occupation"].unique())),
    "relationship": sorted(list(train_dev_data["relationship"].unique())),
    "race": sorted(list(train_dev_data["race"].unique())),
    "gender": sorted(list(train_dev_data["gender"].unique())),
    "native_country": sorted(list(train_dev_data["native_country"].unique())),
}

# All features names.
FEATURE_NAMES = NUMERIC_FEATURE_NAMES + list(
    CATEGORICAL_FEATURES_WITH_VOCABULARY.keys()
)
# Feature default values.
COLUMN_DEFAULTS = [
    [0.0]
    if feature_name in NUMERIC_FEATURE_NAMES + [TARGET_FEATURE_NAME, WEIGHT_COLUMN_NAME]
    else ["NA"]
    for feature_name in CSV_HEADER
]

# 创建一个tf.data.Dataset用于培训和评估
# 我们创建一个输入函数来读取和解析文件，并将特征和标签转换为[ tf.data.Dataset](https://www.tensorflow.org/api_docs/python/tf/data/Dataset)用于训练和评估。

from tensorflow.keras.layers import StringLookup


def process(features, target):
    for feature_name in features:
        if feature_name in CATEGORICAL_FEATURES_WITH_VOCABULARY:
            # Cast categorical feature values to string.
            features[feature_name] = tf.cast(features[feature_name], tf.dtypes.string)
    # Get the instance weight.
    weight = features.pop(WEIGHT_COLUMN_NAME)
    return features, target, weight


def get_dataset_from_csv(csv_file_path, shuffle=False, batch_size=128):

    dataset = tf.data.experimental.make_csv_dataset(
        csv_file_path,
        batch_size=batch_size,
        column_names=CSV_HEADER,
        column_defaults=COLUMN_DEFAULTS,
        label_name=TARGET_FEATURE_NAME,
        num_epochs=1,
        header=False,
        shuffle=shuffle,
    ).map(process)

    return dataset


# 创建模型输入
def create_model_inputs():
    inputs = {}
    for feature_name in FEATURE_NAMES:
        if feature_name in NUMERIC_FEATURE_NAMES:
            inputs[feature_name] = layers.Input(
                name=feature_name, shape=(), dtype=tf.float32
            )
        else:
            inputs[feature_name] = layers.Input(
                name=feature_name, shape=(), dtype=tf.string
            )
    return inputs


# 编码输入特征
# 对于分类特征，我们layers.Embedding使用 encoding_size作为嵌入维度对它们进行编码。
# 对于数值特征，我们使用线性变换将layers.Dense每个特征投影到 encoding_size维向量中。
# 因此，所有编码的特征将具有相同的维度。

def encode_inputs(inputs, encoding_size):
    encoded_features = []
    for feature_name in inputs:
        if feature_name in CATEGORICAL_FEATURES_WITH_VOCABULARY:
            vocabulary = CATEGORICAL_FEATURES_WITH_VOCABULARY[feature_name]
            # Create a lookup to convert a string values to an integer indices.
            # Since we are not using a mask token nor expecting any out of vocabulary
            # (oov) token, we set mask_token to None and  num_oov_indices to 0.
            index = StringLookup(
                vocabulary=vocabulary, mask_token=None, num_oov_indices=1
            )
            # Convert the string input values into integer indices.
            value_index = index(inputs[feature_name])
            # Create an embedding layer with the specified dimensions
            embedding_ecoder = layers.Embedding(
                input_dim=len(vocabulary), output_dim=encoding_size
            )
            # Convert the index values to embedding representations.
            encoded_feature = embedding_ecoder(value_index)
        else:
            # Project the numeric feature to encoding_size using linear transformation.
            encoded_feature = tf.expand_dims(inputs[feature_name], -1)
            encoded_feature = layers.Dense(units=encoding_size)(encoded_feature)
        encoded_features.append(encoded_feature)
    return encoded_features

# 实现门控线性单元
# 门控线性单元 (GLU)提供了抑制与给定任务无关的输入的灵活性。

class GatedLinearUnit(layers.Layer):
    def __init__(self, units):
        super(GatedLinearUnit, self).__init__()
        self.linear = layers.Dense(units)
        self.sigmoid = layers.Dense(units, activation="sigmoid")

    def call(self, inputs):
        return self.linear(inputs) * self.sigmoid(inputs)

# 实施门控残差网络
# 门控残差网络 (GRN) 的工作原理如下：
#
# 将非线性 ELU 变换应用于输入。
# 应用线性变换，然后是 dropout。
# 应用 GLU 并将原始输入添加到 GLU 的输出以执行跳过（残差）连接。
# 应用层标准化并产生输出。
class GatedResidualNetwork(layers.Layer):
    def __init__(self, units, dropout_rate):
        super(GatedResidualNetwork, self).__init__()
        self.units = units
        self.elu_dense = layers.Dense(units, activation="elu")
        self.linear_dense = layers.Dense(units)
        self.dropout = layers.Dropout(dropout_rate)
        self.gated_linear_unit = GatedLinearUnit(units)
        self.layer_norm = layers.LayerNormalization()
        self.project = layers.Dense(units)

    def call(self, inputs):
        x = self.elu_dense(inputs)
        x = self.linear_dense(x)
        x = self.dropout(x)
        if inputs.shape[-1] != self.units:
            inputs = self.project(inputs)
        x = inputs + self.gated_linear_unit(x)
        x = self.layer_norm(x)
        return x


# 实现变量选择网络
# 变量选择网络（VSN）的工作原理如下：
#
# 将 GRN 单独应用于每个功能。
# 对所有特征的串联应用 GRN，然后是 softmax 以产生特征权重。
# 生成单个 GRN 输出的加权和。
# 请注意，无论输入特征的数量如何，VSN 的输出都是 [batch_size, encoding_size]。

class VariableSelection(layers.Layer):
    def __init__(self, num_features, units, dropout_rate):
        super(VariableSelection, self).__init__()
        self.grns = list()
        # Create a GRN for each feature independently
        for idx in range(num_features):
            grn = GatedResidualNetwork(units, dropout_rate)
            self.grns.append(grn)
        # Create a GRN for the concatenation of all the features
        self.grn_concat = GatedResidualNetwork(units, dropout_rate)
        self.softmax = layers.Dense(units=num_features, activation="softmax")

    def call(self, inputs):
        v = layers.concatenate(inputs)
        v = self.grn_concat(v)
        v = tf.expand_dims(self.softmax(v), axis=-1)

        x = []
        for idx, input in enumerate(inputs):
            x.append(self.grns[idx](input))
        x = tf.stack(x, axis=1)

        outputs = tf.squeeze(tf.matmul(v, x, transpose_a=True), axis=1)
        return outputs


# 创建门控残差和变量选择网络模型
def create_model(encoding_size):
    inputs = create_model_inputs()
    feature_list = encode_inputs(inputs, encoding_size)
    num_features = len(feature_list)

    features = VariableSelection(num_features, encoding_size, dropout_rate)(
        feature_list
    )

    outputs = layers.Dense(units=1, activation="sigmoid")(features)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


# 编译、训练和评估模型
learning_rate = 0.001
dropout_rate = 0.15
batch_size = 265
num_epochs = 20
encoding_size = 16

model = create_model(encoding_size)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
    loss=keras.losses.BinaryCrossentropy(),
    metrics=[keras.metrics.BinaryAccuracy(name="accuracy")],
)


# Create an early stopping callback.
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=True
)

print("Start training the model...")
train_dataset = get_dataset_from_csv(
    train_data_file, shuffle=True, batch_size=batch_size
)
valid_dataset = get_dataset_from_csv(valid_data_file, batch_size=batch_size)
model.fit(
    train_dataset,
    epochs=num_epochs,
    validation_data=valid_dataset,
    callbacks=[early_stopping],
)
print("Model training finished.")

print("Evaluating model performance...")
test_dataset = get_dataset_from_csv(test_data_file, batch_size=batch_size)
_, accuracy = model.evaluate(test_dataset)
print(f"Test accuracy: {round(accuracy * 100, 2)}%")

# 62/62 [==============================] - 1s 7ms/step - loss: 60530.6328 - accuracy: 0.8474
# Test accuracy: 84.74%
# 您应该在测试集上达到 84.74% 以上的准确率。

# 要增加模型的学习能力，可以尝试增加 encoding_size值，或者在 VSN 层之上堆叠多个 GRN 层。这可能还需要增加dropout_rate值以避免过度拟合。

def main():
    pass


if __name__ == '__main__':
    main()

