#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# https://keras.io/examples/structured_data/tabtransformer/

import os
import math
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
import matplotlib.pyplot as plt

# 准备数据
# 数据来源：https://archive.ics.uci.edu/ml/machine-learning-databases/adult/
# 此示例使用 UC Irvine Machine Learning Repository 提供的 美国人口普查收入数据集。该任务是二元分类来预测一个人是否有可能年收入超过 50,000 美元。
#
# 该数据集包括 48,842 个实例，具有 14 个输入特征：5 个数字特征和 9 个分类特征。
#
# 首先，让我们将 UCI 机器学习存储库中的数据集加载到 Pandas DataFrame 中：
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



train_data_url = rf"D:\Users\{USERNAME}\data\adult\adult.data"
train_data = pd.read_csv(train_data_url, header=None, names=CSV_HEADER)

test_data_url = rf"D:\Users\{USERNAME}\data\adult\adult.test"
test_data = pd.read_csv(test_data_url, header=None, names=CSV_HEADER)

print(f"Train dataset shape: {train_data.shape}")
print(f"Test dataset shape: {test_data.shape}")

# 删除第一条记录（因为它不是有效的数据示例）和类标签中的尾随“点”。

test_data = test_data[1:]
test_data.income_bracket = test_data.income_bracket.apply(
    lambda value: value.replace(".", "")
)
#

train_data_file = rf"D:\Users\{USERNAME}\data\adult\train_data.csv"
test_data_file = rf"D:\Users\{USERNAME}\data\adult\test_data.csv"

train_data.to_csv(train_data_file, index=False, header=False)
test_data.to_csv(test_data_file, index=False, header=False)

# 定义数据集元数据
# 在这里，我们定义了数据集的元数据，这些元数据对于读取数据并将其解析为输入特征以及根据输入特征的类型对输入特征进行编码很有用。

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
    "workclass": sorted(list(train_data["workclass"].unique())),
    "education": sorted(list(train_data["education"].unique())),
    "marital_status": sorted(list(train_data["marital_status"].unique())),
    "occupation": sorted(list(train_data["occupation"].unique())),
    "relationship": sorted(list(train_data["relationship"].unique())),
    "race": sorted(list(train_data["race"].unique())),
    "gender": sorted(list(train_data["gender"].unique())),
    "native_country": sorted(list(train_data["native_country"].unique())),
}
# 用作实例权重的列的名称。Name of the column to be used as instances weight.
WEIGHT_COLUMN_NAME = "fnlwgt"

# 类别特征（categorical feature）名称列表：
CATEGORICAL_FEATURE_NAMES = list(CATEGORICAL_FEATURES_WITH_VOCABULARY.keys())

# 所有输入特征列表
FEATURE_NAMES = NUMERIC_FEATURE_NAMES + CATEGORICAL_FEATURE_NAMES

# 每个特征的默认值, 默认值的数据类型，会影响最终读取的数据类型；
COLUMN_DEFAULTS = [
    [0.0] if feature_name in NUMERIC_FEATURE_NAMES + [WEIGHT_COLUMN_NAME] else ["NA"]
    for feature_name in CSV_HEADER
]

# 目标特征（target feature）名称：
TARGET_FEATURE_NAME = "income_bracket"

# 目标特征的标签列表
TARGET_LABELS = [" <=50K", " >50K"]

# 配置超参数
# 超参数包括模型架构和训练配置。

LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
DROPOUT_RATE = 0.2
BATCH_SIZE = 265
NUM_EPOCHS = 15

NUM_TRANSFORMER_BLOCKS = 3  # transformer 模块数量.
NUM_HEADS = 4  # 注意力头(attention heads)的数量
EMBEDDING_DIMS = 16  # 分类特征（categorical features）的向量维度（Embedding dimensions）
MLP_HIDDEN_UNITS_FACTORS = [
    2,
    1,
]  # MLP（也称为全连接神经网络）隐藏层单元数(MLP hidden layer units)
NUM_MLP_BLOCKS = 2  # 基线模型MLP模块数量


# 实现数据读取管道
# 我们定义了一个输入函数，它读取和解析文件，然后将特征和标签转换为tf.data.Dataset 用于训练或评估的。

target_label_lookup = layers.StringLookup(
    vocabulary=TARGET_LABELS, mask_token=None, num_oov_indices=0
)


def prepare_example(features, target):
    target_index = target_label_lookup(target)
    weights = features.pop(WEIGHT_COLUMN_NAME)
    return features, target_index, weights


def get_dataset_from_csv(csv_file_path, batch_size=128, shuffle=False):
    dataset = tf.data.experimental.make_csv_dataset(
        csv_file_path,
        batch_size=batch_size,
        column_names=CSV_HEADER,
        column_defaults=COLUMN_DEFAULTS,
        label_name=TARGET_FEATURE_NAME,
        num_epochs=1,
        header=False,
        na_value="?",
        shuffle=shuffle,
    ).map(prepare_example, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
    return dataset.cache()

# 训练及评估
def run_experiment(
    model,
    train_data_file,
    test_data_file,
    num_epochs,
    learning_rate,
    weight_decay,
    batch_size,
):

    optimizer = tfa.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    )

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[keras.metrics.BinaryAccuracy(name="accuracy")],
    )

    train_dataset = get_dataset_from_csv(train_data_file, batch_size, shuffle=True)
    validation_dataset = get_dataset_from_csv(test_data_file, batch_size)

    print("Start training the model...")
    history = model.fit(
        train_dataset, epochs=num_epochs, validation_data=validation_dataset
    )
    print("Model training finished")

    _, accuracy = model.evaluate(validation_dataset, verbose=0)

    print(f"Validation accuracy: {round(accuracy * 100, 2)}%")

    return history


# 创建模型输入
# 现在，将模型的输入定义为字典，其中键是特征名称，值是keras.layers.Input具有相应特征形状和数据类型的张量。

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


# 编码特征
# 该encode_inputs方法返回encoded_categorical_feature_list和numerical_feature_list。
# 我们将分类特征编码为嵌入，对所有特征使用固定embedding_dims值，而不管它们的词汇量大小。
# 这是 Transformer 模型所必需的。

def encode_inputs(inputs, embedding_dims):

    encoded_categorical_feature_list = []
    numerical_feature_list = []

    for feature_name in inputs:
        if feature_name in CATEGORICAL_FEATURE_NAMES:

            # Get the vocabulary of the categorical feature.
            vocabulary = CATEGORICAL_FEATURES_WITH_VOCABULARY[feature_name]

            # Create a lookup to convert string values to an integer indices.
            # Since we are not using a mask token nor expecting any out of vocabulary
            # (oov) token, we set mask_token to None and  num_oov_indices to 0.
            lookup = layers.StringLookup(
                vocabulary=vocabulary,
                mask_token=None,
                num_oov_indices=0,
                output_mode="int",
            )

            # Convert the string input values into integer indices.
            encoded_feature = lookup(inputs[feature_name])

            # Create an embedding layer with the specified dimensions.
            embedding = layers.Embedding(
                input_dim=len(vocabulary), output_dim=embedding_dims
            )

            # Convert the index values to embedding representations.
            encoded_categorical_feature = embedding(encoded_feature)
            encoded_categorical_feature_list.append(encoded_categorical_feature)

        else:

            # Use the numerical features as-is.
            numerical_feature = tf.expand_dims(inputs[feature_name], -1)
            numerical_feature_list.append(numerical_feature)

    return encoded_categorical_feature_list, numerical_feature_list


# 实现 MLP 块
def create_mlp(hidden_units, dropout_rate, activation, normalization_layer, name=None):

    mlp_layers = []
    for units in hidden_units:
        mlp_layers.append(normalization_layer),
        mlp_layers.append(layers.Dense(units, activation=activation))
        mlp_layers.append(layers.Dropout(dropout_rate))

    return keras.Sequential(mlp_layers, name=name)


# 实验 1：基线模型
# 在第一个实验中，我们创建了一个简单的多层前馈网络。

def create_baseline_model(
    embedding_dims, num_mlp_blocks, mlp_hidden_units_factors, dropout_rate
):

    # Create model inputs.
    inputs = create_model_inputs()
    # encode features.
    encoded_categorical_feature_list, numerical_feature_list = encode_inputs(
        inputs, embedding_dims
    )
    # Concatenate all features.
    features = layers.concatenate(
        encoded_categorical_feature_list + numerical_feature_list
    )
    # Compute Feedforward layer units.
    feedforward_units = [features.shape[-1]]

    # Create several feedforwad layers with skip connections.
    for layer_idx in range(num_mlp_blocks):
        features = create_mlp(
            hidden_units=feedforward_units,
            dropout_rate=dropout_rate,
            activation=keras.activations.gelu,
            normalization_layer=layers.LayerNormalization(epsilon=1e-6),
            name=f"feedforward_{layer_idx}",
        )(features)

    # Compute MLP hidden_units.
    mlp_hidden_units = [
        factor * features.shape[-1] for factor in mlp_hidden_units_factors
    ]
    # Create final MLP.
    features = create_mlp(
        hidden_units=mlp_hidden_units,
        dropout_rate=dropout_rate,
        activation=keras.activations.selu,
        normalization_layer=layers.BatchNormalization(),
        name="MLP",
    )(features)

    # Add a sigmoid as a binary classifer.
    outputs = layers.Dense(units=1, activation="sigmoid", name="sigmoid")(features)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


baseline_model = create_baseline_model(
    embedding_dims=EMBEDDING_DIMS,
    num_mlp_blocks=NUM_MLP_BLOCKS,
    mlp_hidden_units_factors=MLP_HIDDEN_UNITS_FACTORS,
    dropout_rate=DROPOUT_RATE,
)

print("Total model weights:", baseline_model.count_params())
keras.utils.plot_model(baseline_model, show_shapes=True, rankdir="LR")
# Total model weights: 109629
# PNG

# 让我们训练和评估基线模型：

history = run_experiment(
    model=baseline_model,
    train_data_file=train_data_file,
    test_data_file=test_data_file,
    num_epochs=NUM_EPOCHS,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    batch_size=BATCH_SIZE,
)

# Epoch 15/15
# 123/123 [==============================] - 2s 16ms/step - loss: 62646.2344 - accuracy: 0.8337 - val_loss: 62280.1094 - val_accuracy: 0.8343
# Model training finished
# Validation accuracy: 83.43%
# 基线线性模型实现了约 83.4% 的验证准确度。

# 实验 2：TabTransformer
# TabTransformer 架构的工作原理如下：
#
# 所有分类特征都被编码为嵌入，使用相同的embedding_dims. 这意味着每个分类特征中的每个值都有自己的嵌入向量。
# 将列嵌入（每个分类特征的一个嵌入向量）添加（逐点）到分类特征嵌入中。
# 嵌入的分类特征被输入到一堆 Transformer 块中。每个 Transformer 块由一个多头自注意力层和一个前馈层组成。
# 最终 Transformer 层的输出，即分类特征的上下文嵌入，与输入数字特征连接，并馈入最终的 MLP 块。
# 在模型的softmax末尾应用分类器。
# 本文在附录：实验和模型详细信息部分讨论了列嵌入的添加和连接 。TabTransformer 的架构如下所示，如论文中所述。



def create_tabtransformer_classifier(
    num_transformer_blocks,
    num_heads,
    embedding_dims,
    mlp_hidden_units_factors,
    dropout_rate,
    use_column_embedding=False,
):

    # Create model inputs.
    inputs = create_model_inputs()
    # encode features.
    encoded_categorical_feature_list, numerical_feature_list = encode_inputs(
        inputs, embedding_dims
    )
    # Stack categorical feature embeddings for the Tansformer.
    encoded_categorical_features = tf.stack(encoded_categorical_feature_list, axis=1)
    # Concatenate numerical features.
    numerical_features = layers.concatenate(numerical_feature_list)

    # Add column embedding to categorical feature embeddings.
    if use_column_embedding:
        num_columns = encoded_categorical_features.shape[1]
        column_embedding = layers.Embedding(
            input_dim=num_columns, output_dim=embedding_dims
        )
        column_indices = tf.range(start=0, limit=num_columns, delta=1)
        encoded_categorical_features = encoded_categorical_features + column_embedding(
            column_indices
        )

    # Create multiple layers of the Transformer block.
    for block_idx in range(num_transformer_blocks):
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embedding_dims,
            dropout=dropout_rate,
            name=f"multihead_attention_{block_idx}",
        )(encoded_categorical_features, encoded_categorical_features)
        # Skip connection 1.
        x = layers.Add(name=f"skip_connection1_{block_idx}")(
            [attention_output, encoded_categorical_features]
        )
        # Layer normalization 1.
        x = layers.LayerNormalization(name=f"layer_norm1_{block_idx}", epsilon=1e-6)(x)
        # Feedforward.
        feedforward_output = create_mlp(
            hidden_units=[embedding_dims],
            dropout_rate=dropout_rate,
            activation=keras.activations.gelu,
            normalization_layer=layers.LayerNormalization(epsilon=1e-6),
            name=f"feedforward_{block_idx}",
        )(x)
        # Skip connection 2.
        x = layers.Add(name=f"skip_connection2_{block_idx}")([feedforward_output, x])
        # Layer normalization 2.
        encoded_categorical_features = layers.LayerNormalization(
            name=f"layer_norm2_{block_idx}", epsilon=1e-6
        )(x)

    # Flatten the "contextualized" embeddings of the categorical features.
    categorical_features = layers.Flatten()(encoded_categorical_features)
    # Apply layer normalization to the numerical features.
    numerical_features = layers.LayerNormalization(epsilon=1e-6)(numerical_features)
    # Prepare the input for the final MLP block.
    features = layers.concatenate([categorical_features, numerical_features])

    # Compute MLP hidden_units.
    mlp_hidden_units = [
        factor * features.shape[-1] for factor in mlp_hidden_units_factors
    ]
    # Create final MLP.
    features = create_mlp(
        hidden_units=mlp_hidden_units,
        dropout_rate=dropout_rate,
        activation=keras.activations.selu,
        normalization_layer=layers.BatchNormalization(),
        name="MLP",
    )(features)

    # Add a sigmoid as a binary classifer.
    outputs = layers.Dense(units=1, activation="sigmoid", name="sigmoid")(features)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


tabtransformer_model = create_tabtransformer_classifier(
    num_transformer_blocks=NUM_TRANSFORMER_BLOCKS,
    num_heads=NUM_HEADS,
    embedding_dims=EMBEDDING_DIMS,
    mlp_hidden_units_factors=MLP_HIDDEN_UNITS_FACTORS,
    dropout_rate=DROPOUT_RATE,
)

print("Total model weights:", tabtransformer_model.count_params())
keras.utils.plot_model(tabtransformer_model, show_shapes=True, rankdir="LR")
# Total model weights: 87479
# PNG

# 让我们训练和评估 TabTransformer 模型：

history = run_experiment(
    model=tabtransformer_model,
    train_data_file=train_data_file,
    test_data_file=test_data_file,
    num_epochs=NUM_EPOCHS,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    batch_size=BATCH_SIZE,
)
# 123/123 [==============================] - 9s 73ms/step - loss: 61024.8984 - accuracy: 0.8452 - val_loss: 60986.0703 - val_accuracy: 0.8449
# Model training finished
# Validation accuracy: 84.49%
# TabTransformer 模型实现了约 84.49% 的验证准确度。请注意，在默认参数配置下，baseline 和 TabTransformer 的可训练权重数量相似：分别为 109,629 和 92,151，并且都使用相同的训练超参数。

# 结论
# TabTransformer 在表格数据方面明显优于 MLP 和最近的深度网络，同时与基于树的集成模型的性能相匹配。
# TabTransformer 可以使用标记示例在端到端的监督训练中学习。
# 对于有少量标记示例和大量未标记示例的场景，可以采用预训练过程来使用未标记数据训练 Transformer 层。
# 接下来是使用标记数据对预训练的 Transformer 层以及顶部 MLP 层进行微调。

def main():
    pass


if __name__ == '__main__':
    main()