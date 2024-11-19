#!/usr/bin/env python
# coding=utf-8

# 来源：https://keras.io/examples/vision/metric_learning/

import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import random
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from collections import defaultdict
from PIL import Image
from sklearn.metrics import ConfusionMatrixDisplay
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import RepeatVector, Permute, Reshape
from tensorflow.keras import backend as K
from tensorflow.keras.datasets import cifar10, mnist

from tensorflow.keras.utils import plot_model


import os
USERNAME = os.getenv("USERNAME")
import pandas as pd
from sklearn.utils import shuffle
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Embedding, LSTM, Dropout, Input, BatchNormalization, Lambda, Flatten
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Layer
from tensorflow.keras import regularizers
from tensorflow.keras.regularizers import l2
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, roc_curve

# 交叉熵损失函数
from tensorflow.python.ops.nn_ops import softmax_cross_entropy_with_logits_v2
from tensorflow.python.ops.nn_ops import sparse_softmax_cross_entropy_with_logits_v2

############################################################################################################################

data_file = rf"D:\Users\{USERNAME}/github_project/toutiao-text-classfication-dataset/toutiao_cat_data.txt"  # 数据来源： http://github.com/skdjfla/toutiao-text-classfication-dataset
df = pd.read_csv(data_file, sep='_!_', names=['id', 'code', 'label', 'title', 'keyword'], engine='python', usecols=['title', 'keyword', 'label'])
df = shuffle(df)
df = df.fillna('')
texts = [' '.join(list(text + keyword.replace(',', ''))) for text, keyword in df[['title', 'keyword']].values]
tokenizer = Tokenizer(num_words=8000)
# 根据文本列表更新内部词汇表。
tokenizer.fit_on_texts(texts)

# 将文本中的每个文本转换为整数序列。
# 只有最常出现的“num_words”字才会被考虑在内。
# 只有分词器知道的单词才会被考虑在内。
sequences = tokenizer.texts_to_sequences(texts)
# dict {word: index}
word_index = tokenizer.word_index

print('tokens数量：', len(word_index))
maxlen=48
data = pad_sequences(sequences, maxlen=maxlen)
print('Shape of data tensor:', data.shape)

code2id = {label: idx for idx, label in enumerate(sorted(df['label'].unique()))}
y = np.array([code2id[label] for label in df['label'].values])
num_classes = len(code2id) # 15类

del texts, sequences

x_train, x_test, y_train, y_test = train_test_split(data, y, test_size=0.2)
# x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5)

############################################################################################################################

# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# 
# x_train = x_train.astype("float32") / 255.0
# y_train = np.squeeze(y_train)
# x_test = x_test.astype("float32") / 255.0
# y_test = np.squeeze(y_test)

class_idx_to_train_idxs = defaultdict(list)
for y_train_idx, y in enumerate(y_train):
    class_idx_to_train_idxs[y].append(y_train_idx)

class_idx_to_test_idxs = defaultdict(list)
for y_test_idx, y in enumerate(y_test):
    class_idx_to_test_idxs[y].append(y_test_idx)

# 
# height_width = 28

class AnchorPositivePairs(keras.utils.Sequence):
    def __init__(self, num_batches):
        super().__init__()
        self.num_batches = num_batches

    def __len__(self):
        return self.num_batches

    def __getitem__(self, _idx):
        x = np.empty((2, num_classes, maxlen), dtype=np.float32)
        for class_idx in range(num_classes):
            examples_for_class = class_idx_to_train_idxs[class_idx]
            anchor_idx = random.choice(examples_for_class)
            positive_idx = random.choice(examples_for_class)
            while positive_idx == anchor_idx:
                positive_idx = random.choice(examples_for_class)
            x[0, class_idx] = x_train[anchor_idx]
            x[1, class_idx] = x_train[positive_idx]
        return x

class EmbeddingModel(keras.Model):
    def train_step(self, data):
        # Note: Workaround for open issue, to be removed.
        if isinstance(data, tuple):
            data = data[0]
        anchors, positives = data[0], data[1]

        with tf.GradientTape() as tape:
            # Run both anchors and positives through model.
            anchor_embeddings = self(anchors, training=True)
            positive_embeddings = self(positives, training=True)

            # Calculate cosine similarity between anchors and positives. As they have
            # been normalised this is just the pair wise dot products.
            similarities = tf.einsum(
                "ae,pe->ap", anchor_embeddings, positive_embeddings
            )

            # Since we intend to use these as logits we scale them by a temperature.
            # This value would normally be chosen as a hyper parameter.
            temperature = 0.2
            similarities /= temperature

            # We use these similarities as logits for a softmax. The labels for
            # this call are just the sequence [0, 1, 2, ..., num_classes] since we
            # want the main diagonal values, which correspond to the anchor/positive
            # pairs, to be high. This loss will move embeddings for the
            # anchor/positive pairs together and move all other pairs apart.
            sparse_labels = K.arange(num_classes)
            loss = self.compute_loss(y=sparse_labels, y_pred=similarities)

        # Calculate gradients and apply via optimizer.
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Update and return metrics (specifically the one for the loss value).
        for metric in self.metrics:
            # Calling `self.compile` will by default add a [`keras.metrics.Mean`](/api/metrics/metrics_wrappers#mean-class) loss
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(sparse_labels, similarities)

        return {m.name: m.result() for m in self.metrics}

inputs = layers.Input(shape=(maxlen,))
x = Embedding(8000, 32)(inputs)
x = LSTM(256, dropout=0.2, recurrent_dropout=0.2)(x)
# x = Dropout(0.2)(x)
x = Dense(256, activation="relu")(x)
embeddings = layers.Dense(units=8, activation=None)(x)

embeddings = layers.UnitNormalization()(embeddings)

model = EmbeddingModel(inputs, embeddings)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
)

history = model.fit(AnchorPositivePairs(num_batches=1000), epochs=10)

plt.plot(history.history["loss"])
plt.show()

# print(history.history)
# {'loss': [2.4982059001922607, 2.0010366439819336, 1.7315436601638794, 1.5839107036590576, 1.4166194200515747, 1.3148120641708374, 1.2170499563217163, 1.1300288438796997, 1.1105986833572388, 1.0604097843170166, 1.034447193145752, 0.9938744902610779, 0.9700401425361633, 0.9406204223632812, 0.9524050354957581, 0.9155259132385254, 0.9237129092216492, 0.8836346864700317, 0.8812150359153748, 0.8604253530502319]}

#############################################################################################################################
# 训练的模型效果展示：
near_neighbours_per_example = 10

embeddings = model.predict(x_test)
embeddings = embeddings.astype(np.float16)

gram_matrix = np.einsum("ae,be->ab", embeddings, embeddings)
near_neighbours = np.argsort(gram_matrix.T)[:, -(near_neighbours_per_example + 1) :]  # 取每个样例数据最相近的 10 个数据

confusion_matrix = np.zeros((num_classes, num_classes))

# For each class.
for class_idx in range(num_classes):
    # Consider 10 examples.
    example_idxs = class_idx_to_test_idxs[class_idx][:10]  # 同组的10个数据
    for y_test_idx in example_idxs:
        # And count the classes of its near neighbours.
        for nn_idx in near_neighbours[y_test_idx][:-1]:  # 最相近的索引, 及排除最相似的一个，因为这个是自己；
            nn_class_idx = y_test[nn_idx]  # 根据索引获取标签
            confusion_matrix[class_idx, nn_class_idx] += 1  # 计数

# Display a confusion matrix.
labels = [str(i) for i in range(num_classes)]
disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=labels)
disp.plot(include_values=True, cmap="viridis", ax=None, xticks_rotation="vertical")
plt.show()

# confusion_matrix
# Out[24]: 
# [[80.  0.  1.  0.  2.  2.  0.  9.  0.  0.  0.  0.  6.  0.  0.]
#  [ 0. 75.  0.  0.  1.  3.  0.  0.  0.  0.  0. 17.  4.  0.  0.]
#  [ 0.  0. 67.  1.  3.  5.  7.  1.  0.  0.  0.  5. 10.  1.  0.]
#  [ 1.  3.  2. 71.  1. 16.  1.  0.  0.  1.  0.  4.  0.  0.  0.]
#  [ 0.  0.  3.  4. 60.  2.  0.  0.  8. 18.  2.  2.  1.  0.  0.]
#  [ 1.  1.  9.  1.  0. 50.  1.  2.  0.  0.  0. 34.  1.  0.  0.]
#  [ 0.  0.  8.  0.  9.  0. 62.  0.  3. 15.  0.  2.  0.  1.  0.]
#  [ 0.  1.  0.  0.  5.  6.  0. 85.  0.  0.  0.  2.  1.  0.  0.]
#  [ 9.  0.  0.  1.  2.  0.  0.  0. 68.  8.  0.  2.  0. 10.  0.]
#  [ 0.  0.  0.  0.  0.  0.  0.  0.  0. 99.  0.  0.  0.  1.  0.]
#  [ 1.  1.  3.  1. 24.  2.  0.  5.  0.  0. 60.  0.  0.  3.  0.]
#  [ 0.  2.  0.  0.  0. 19.  0.  0.  0.  0.  0. 78.  0.  1.  0.]
#  [16.  9.  6.  9.  1.  3.  0.  1.  1.  0.  0.  1. 45.  8.  0.]
#  [ 0.  0.  0.  0.  6.  1.  0.  0. 23. 11.  1.  0.  1. 57.  0.]
#  [ 3.  0.  0.  0.  0. 70.  0.  0.  0.  0.  0.  3.  0.  1.  3.]]

num = 0
for y_test_idx in range(near_neighbours.shape[0]):
    if len([1 for nn_idx in near_neighbours[y_test_idx][:-1] if y_test[y_test_idx] == y_test[nn_idx]])>5:
        num += 1
print("正确率：", num/near_neighbours.shape[0])
# 正确率： 0.8017


#############################################################################################################################
# 以此相似搜索模型为基础，并锁定相关参数，进一步训练文本分类模型：

for l in model.layers:
    l.trainable = False

keras.backend.clear_session()

x = Dense(64, activation="relu", name="dense_6")(model.output)
x = Dropout(0.2)(x)
outputs = Dense(num_classes, activation='softmax', name='output')(x)
cls_model = Model(model.input, outputs)
cls_model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])

print(cls_model.summary())

# 训练模型
history = cls_model.fit(x_train, y_train, epochs=10, validation_split=0.2, batch_size=64)
print(history.history)

plt.plot(history.history["loss"])
plt.show()

# Epoch 1/10
# 3827/3827 [==============================] - 487s 127ms/step - loss: 0.6807 - accuracy: 0.8315 - val_loss: 0.5434 - val_accuracy: 0.8530
# Epoch 2/10
# 3827/3827 [==============================] - 709s 185ms/step - loss: 0.5867 - accuracy: 0.8448 - val_loss: 0.5367 - val_accuracy: 0.8538
# Epoch 3/10
# 3827/3827 [==============================] - 576s 150ms/step - loss: 0.5805 - accuracy: 0.8457 - val_loss: 0.5319 - val_accuracy: 0.8546
# Epoch 4/10
# 3827/3827 [==============================] - 343s 90ms/step - loss: 0.5749 - accuracy: 0.8458 - val_loss: 0.5289 - val_accuracy: 0.8551
# Epoch 5/10
# 3827/3827 [==============================] - 341s 89ms/step - loss: 0.5723 - accuracy: 0.8454 - val_loss: 0.5262 - val_accuracy: 0.8554
# Epoch 6/10
# 3827/3827 [==============================] - 340s 89ms/step - loss: 0.5705 - accuracy: 0.8454 - val_loss: 0.5249 - val_accuracy: 0.8553
# Epoch 7/10
# 3827/3827 [==============================] - 343s 90ms/step - loss: 0.5682 - accuracy: 0.8451 - val_loss: 0.5242 - val_accuracy: 0.8548
# Epoch 8/10
# 3827/3827 [==============================] - 359s 94ms/step - loss: 0.5667 - accuracy: 0.8461 - val_loss: 0.5235 - val_accuracy: 0.8551
# Epoch 9/10
# 3827/3827 [==============================] - 369s 97ms/step - loss: 0.5657 - accuracy: 0.8457 - val_loss: 0.5220 - val_accuracy: 0.8548
# Epoch 10/10
# 3827/3827 [==============================] - 540s 141ms/step - loss: 0.5638 - accuracy: 0.8461 - val_loss: 0.5216 - val_accuracy: 0.8540
# {'loss': [0.6807172894477844, 0.5866665840148926, 0.5804747939109802, 0.5748817920684814, 0.5723122358322144, 0.5704941749572754, 0.5681867003440857, 0.5667027235031128, 0.5657093524932861, 0.5638495087623596], 'accuracy': [0.8314592242240906, 0.8447983264923096, 0.8456965684890747, 0.8458353877067566, 0.8453739881515503, 0.8454352617263794, 0.8451331257820129, 0.8461456894874573, 0.8457047343254089, 0.8461089134216309], 'val_loss': [0.5434331297874451, 0.5366917848587036, 0.5319305062294006, 0.528885006904602, 0.526159405708313, 0.5248529314994812, 0.5241885185241699, 0.5234854221343994, 0.5219812989234924, 0.5215981006622314], 'val_accuracy': [0.8529968857765198, 0.8538461327552795, 0.8545974493026733, 0.855120062828064, 0.8553977012634277, 0.8552833795547485, 0.8548097610473633, 0.8550710678100586, 0.8548260927200317, 0.854042112827301]}

def main():
    pass


if __name__ == "__main__":
    main()
