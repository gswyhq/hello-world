#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 流程：
# 使用 Pandas 加载 CSV 文件。
# 构建输入流水线以使用 tf.data 对行进行批处理和乱序。
# 使用 Keras 预处理层将 CSV 中的列映射到用于训练模型的特征。
# 使用 Keras 构建、训练和评估模型。

# 资料来源： https://tensorflow.google.cn/tutorials/structured_data/preprocessing_layers?hl=zh-cn

# 使用 Pandas 创建数据帧
# Pandas 是一个 Python 库，其中包含许多有用的加载和处理结构化数据的实用工具。您将使用 Pandas 从 URL 下载数据集，并将其加载到数据帧中。

import pandas as pd
import pathlib
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers.experimental import preprocessing

# dataset_url = 'http://storage.googleapis.com/download.tensorflow.org/data/petfinder-mini.zip'
# csv_file = 'datasets/petfinder-mini/petfinder-mini.csv'
csv_file = '~/Downloads/petfinder-mini/petfinder-mini.csv'

# tf.keras.utils.get_file('petfinder_mini.zip', dataset_url,
#                         extract=True, cache_dir='.')
dataframe = pd.read_csv(csv_file)

# Downloading data from http://storage.googleapis.com/download.tensorflow.org/data/petfinder-mini.zip
# 1668792/1668792 [==============================] - 0s 0us/step

dataframe.head()

# 创建目标变量
# Kaggle 比赛中的任务是预测宠物被领养的速度（例如，在第一周、第一个月、前三个月等）。我们针对教程进行一下简化。在这里，您将把它转化为一个二元分类问题，并简单地预测宠物是否被领养。
#
# 修改标签列后，0 表示宠物未被领养，1 表示宠物已被领养。


# In the original dataset "4" indicates the pet was not adopted.
dataframe['target'] = np.where(dataframe['AdoptionSpeed']==4, 0, 1)

# Drop un-used columns.
dataframe = dataframe.drop(columns=['AdoptionSpeed', 'Description'])

# 将数据帧拆分为训练集、验证集和测试集
# 您下载的数据集是单个 CSV 文件。您将把它拆分为训练集、验证集和测试集。


train, test = train_test_split(dataframe, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)
print(len(train), 'train examples')
print(len(val), 'validation examples')
print(len(test), 'test examples')

# 使用 tf.data 创建输入流水线
# 接下来，您将使用 tf.data 封装数据帧，以便对数据进行乱序和批处理。如果您处理的 CSV 文件非常大（大到无法放入内存），则可以使用 tf.data 直接从磁盘读取文件。本教程中没有涉及这方面的内容。


# A utility method to create a tf.data dataset from a Pandas Dataframe
def df_to_dataset(dataframe, shuffle=True, batch_size=32):
  dataframe = dataframe.copy()
  labels = dataframe.pop('target')
  ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))
  ds = ds.batch(batch_size)
  ds = ds.prefetch(batch_size)
  return ds

# 现在您已经创建了输入流水线，我们调用它来查看它返回的数据的格式。您使用了小批次来保持输出的可读性。

batch_size = 5
train_ds = df_to_dataset(train, batch_size=batch_size)

[(train_features, label_batch)] = train_ds.take(1)
print('Every feature:', list(train_features.keys()))
print('A batch of ages:', train_features['Age'])
print('A batch of targets:', label_batch )

# 您可以看到数据集（从数据帧）返回了一个列名称字典，该字典映射到来自数据帧中行的列值。

# 演示预处理层的使用。
# Keras 预处理层 API 允许您构建 Keras 原生输入处理流水线。您将使用 3 个预处理层来演示特征预处理代码。

# Normalization - 数据的特征归一化。
# Normalization - 类别编码层。
# StringLookup - 将字符串从词汇表映射到整数索引。
# IntegerLookup - 将词汇表中的整数映射到整数索引。
# 您可以在此处找到可用预处理层的列表。

# 数值列
# 对于每个数值特征，您将使用 Normalization() 层来确保每个特征的平均值为 0，且其标准差为 1。

# get_normalization_layer 函数返回一个层，该层将特征归一化应用于数值特征。


def get_normalization_layer(name, dataset):
  # Create a Normalization layer for our feature.
  normalizer = preprocessing.Normalization(axis=None)

  # Prepare a Dataset that only yields our feature.
  feature_ds = dataset.map(lambda x, y: x[name])

  # Learn the statistics of the data.
  normalizer.adapt(feature_ds)

  return normalizer

photo_count_col = train_features['PhotoAmt']
layer = get_normalization_layer('PhotoAmt', train_ds)
layer(photo_count_col)

# Out[21]:
# <tf.Tensor: shape=(5,), dtype=float32, numpy=
# array([-0.5148777, -0.8333284, -0.5148777,  0.7589252,  0.7589252],
#       dtype=float32)>

# 注：如果您有许多数值特征（数百个或更多），首先将它们连接起来并使用单个 normalization 层会更有效。

# 分类列
# 在此数据集中，Type 表示为字符串（例如 'Dog' 或 'Cat'）。您不能将字符串直接馈送给模型。预处理层负责将字符串表示为独热向量。

# get_category_encoding_layer 函数返回一个层，该层将值从词汇表映射到整数索引，并对特征进行独热编码。


def get_category_encoding_layer(name, dataset, dtype, max_tokens=None):
  # Create a StringLookup layer which will turn strings into integer indices
  if dtype == 'string':
    index = preprocessing.StringLookup(max_tokens=max_tokens)
  else:
    index = preprocessing.IntegerLookup(max_tokens=max_tokens)

  # Prepare a Dataset that only yields our feature
  feature_ds = dataset.map(lambda x, y: x[name])

  # Learn the set of possible values and assign them a fixed integer index.
  index.adapt(feature_ds)

  # Create a Discretization for our integer indices.
  encoder = preprocessing.CategoryEncoding(num_tokens=index.vocabulary_size())

  # Apply one-hot encoding to our indices. The lambda function captures the
  # layer so we can use them, or include them in the functional model later.
  return lambda feature: encoder(index(feature))

type_col = train_features['Type']
layer = get_category_encoding_layer('Type', train_ds, 'string')
layer(type_col)

# <tf.Tensor: shape=(3,), dtype=float32, numpy=array([0., 1., 1.], dtype=float32)>
# 通常，您不应将数字直接输入模型，而是改用这些输入的独热编码。考虑代表宠物年龄的原始数据。


type_col = train_features['Age']
category_encoding_layer = get_category_encoding_layer('Age', train_ds,
                                                      'int64', 5)
category_encoding_layer(type_col)

# <tf.Tensor: shape=(5,), dtype=float32, numpy=array([1., 1., 0., 1., 1.], dtype=float32)>
# 选择要使用的列
# 您已经了解了如何使用多种类型的预处理层。现在您将使用它们来训练模型。您将使用 Keras-functional API 来构建模型。Keras 函数式 API 是一种比 tf.keras.Sequential API 更灵活的创建模型的方式。
#
# 本教程的目标是向您展示使用预处理层所需的完整代码（例如机制）。任意选择了几列来训练我们的模型。
#
# 要点：如果您的目标是构建一个准确的模型，请尝试使用自己的更大的数据集，并仔细考虑哪些特征最有意义，以及它们应该如何表示。
#
# 之前，您使用了小批次来演示输入流水线。现在让我们创建一个具有更大批次大小的新输入流水线。


batch_size = 32
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

all_inputs = []
encoded_features = []

# Numeric features.
for header in ['PhotoAmt', 'Fee']:
  numeric_col = tf.keras.Input(shape=(1,), name=header)
  normalization_layer = get_normalization_layer(header, train_ds)
  encoded_numeric_col = normalization_layer(numeric_col)
  all_inputs.append(numeric_col)
  encoded_features.append(encoded_numeric_col)

# Categorical features encoded as integers.
age_col = tf.keras.Input(shape=(1,), name='Age', dtype='int64')
encoding_layer = get_category_encoding_layer('Age', train_ds, dtype='int64',
                                             max_tokens=5)
encoded_age_col = encoding_layer(age_col)
all_inputs.append(age_col)
encoded_features.append(encoded_age_col)

# Categorical features encoded as string.
categorical_cols = ['Type', 'Color1', 'Color2', 'Gender', 'MaturitySize',
                    'FurLength', 'Vaccinated', 'Sterilized', 'Health', 'Breed1']
for header in categorical_cols:
  categorical_col = tf.keras.Input(shape=(1,), name=header, dtype='string')
  encoding_layer = get_category_encoding_layer(header, train_ds, dtype='string',
                                               max_tokens=5)
  encoded_categorical_col = encoding_layer(categorical_col)
  all_inputs.append(categorical_col)
  encoded_features.append(encoded_categorical_col)

# 创建、编译并训练模型
# 接下来，您可以创建端到端模型。


all_features = tf.keras.layers.concatenate(encoded_features)
x = tf.keras.layers.Dense(32, activation="relu")(all_features)
x = tf.keras.layers.Dropout(0.5)(x)
output = tf.keras.layers.Dense(1)(x)
model = tf.keras.Model(all_inputs, output)
model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=["accuracy"])

# 我们来可视化连接图：

# rankdir='LR' is used to make the graph horizontal.
tf.keras.utils.plot_model(model, show_shapes=True, rankdir="LR")

# You must install pydot (`pip install pydot`) and install graphviz (see instructions at https://graphviz.gitlab.io/download/) for plot_model/model_to_dot to work.
# 训练模型。

model.fit(train_ds, epochs=10, validation_data=val_ds)

# Epoch 1/10
# 29/29 [==============================] - 2s 29ms/step - loss: 0.7471 - accuracy: 0.4195 - val_loss: 0.5727 - val_accuracy: 0.6555
# Epoch 2/10
# 29/29 [==============================] - 0s 10ms/step - loss: 0.6032 - accuracy: 0.6352 - val_loss: 0.5402 - val_accuracy: 0.7389
# Epoch 3/10
# 29/29 [==============================] - 0s 10ms/step - loss: 0.5787 - accuracy: 0.6755 - val_loss: 0.5277 - val_accuracy: 0.7497
# Epoch 4/10
# 29/29 [==============================] - 0s 10ms/step - loss: 0.5632 - accuracy: 0.6900 - val_loss: 0.5202 - val_accuracy: 0.7497
# Epoch 5/10
# 29/29 [==============================] - 0s 10ms/step - loss: 0.5536 - accuracy: 0.6931 - val_loss: 0.5154 - val_accuracy: 0.7427
# Epoch 6/10
# 29/29 [==============================] - 0s 10ms/step - loss: 0.5401 - accuracy: 0.7066 - val_loss: 0.5124 - val_accuracy: 0.7476
# Epoch 7/10
# 29/29 [==============================] - 0s 10ms/step - loss: 0.5367 - accuracy: 0.7088 - val_loss: 0.5108 - val_accuracy: 0.7416
# Epoch 8/10
# 29/29 [==============================] - 0s 10ms/step - loss: 0.5352 - accuracy: 0.7112 - val_loss: 0.5087 - val_accuracy: 0.7470
# Epoch 9/10
# 29/29 [==============================] - 0s 10ms/step - loss: 0.5302 - accuracy: 0.7192 - val_loss: 0.5073 - val_accuracy: 0.7449
# Epoch 10/10
# 29/29 [==============================] - 0s 10ms/step - loss: 0.5303 - accuracy: 0.7194 - val_loss: 0.5062 - val_accuracy: 0.7476
# <keras.callbacks.History at 0x7fd8ac630610>

loss, accuracy = model.evaluate(test_ds)
print("Accuracy", accuracy)

# 10/10 [==============================] - 0s 7ms/step - loss: 0.5158 - accuracy: 0.7357
# Accuracy 0.7357019186019897
# 根据新数据进行推断
# 要点：您开发的模型现在可以直接从 CSV 文件中对行进行分类，因为预处理代码包含在模型本身中。
#
# 现在，您可以保存并重新加载 Keras 模型。请按照此处的教程了解有关 TensorFlow 模型的更多信息。


model.save('my_pet_classifier')
reloaded_model = tf.keras.models.load_model('my_pet_classifier')

# WARNING:absl:Function `_wrapped_model` contains input name(s) PhotoAmt, Fee, Age, Type, Color1, Color2, Gender, MaturitySize, FurLength, Vaccinated, Sterilized, Health, Breed1 with unsupported characters which will be renamed to photoamt, fee, age, type, color1, color2, gender, maturitysize, furlength, vaccinated, sterilized, health, breed1 in the SavedModel.
# INFO:tensorflow:Assets written to: my_pet_classifier/assets
# INFO:tensorflow:Assets written to: my_pet_classifier/assets
# 要获得对新样本的预测，只需调用 model.predict()。您只需要做两件事：
#
# 将标量封装成列表，以便具有批次维度（模型只处理成批次的数据，而不是单个样本）
# 对每个特征调用 convert_to_tensor

sample = {
    'Type': 'Cat',
    'Age': 3,
    'Breed1': 'Tabby',
    'Gender': 'Male',
    'Color1': 'Black',
    'Color2': 'White',
    'MaturitySize': 'Small',
    'FurLength': 'Short',
    'Vaccinated': 'No',
    'Sterilized': 'No',
    'Health': 'Healthy',
    'Fee': 100,
    'PhotoAmt': 2,
}

input_dict = {name: tf.convert_to_tensor([value]) for name, value in sample.items()}
predictions = reloaded_model.predict(input_dict)
prob = tf.nn.sigmoid(predictions[0])

print(
    "该宠物被收养概率： %.1f " % (100 * prob)
)

# 1/1 [==============================] - 1s 507ms/step
# This particular pet had a 79.3 percent probability of getting adopted.
# 要点：对于更大、更复杂的数据集，您通常会看到深度学习的最佳结果。在处理像这样的小数据集时，我们建议使用决策树或随机森林作为强基线。本教程的目标是演示处理结构化数据的机制，以便您将来处理自己的数据集时有可以作为起点的代码。
#
# 后续步骤
# 进一步了解有关结构化数据分类的最佳方法是自己尝试。您可能希望找到另一个可使用的数据集，并使用与上述类似的代码训练模型对其进行分类。为了提高准确率，请仔细考虑要在模型中包含哪些特征，以及它们应该如何表示。

def main():
    pass


if __name__ == '__main__':
    main()