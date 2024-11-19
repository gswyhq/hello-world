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

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype("float32") / 255.0
y_train = np.squeeze(y_train)
x_test = x_test.astype("float32") / 255.0
y_test = np.squeeze(y_test)

class_idx_to_train_idxs = defaultdict(list)
for y_train_idx, y in enumerate(y_train):
    class_idx_to_train_idxs[y].append(y_train_idx)

class_idx_to_test_idxs = defaultdict(list)
for y_test_idx, y in enumerate(y_test):
    class_idx_to_test_idxs[y].append(y_test_idx)

num_classes = 10
height_width = 28

class AnchorPositivePairs(keras.utils.Sequence):
    def __init__(self, num_batches):
        super().__init__()
        self.num_batches = num_batches

    def __len__(self):
        return self.num_batches

    def __getitem__(self, _idx):
        x = np.empty((2, num_classes, height_width, height_width), dtype=np.float32)
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

inputs = layers.Input(shape=(height_width, height_width))
x = Reshape(target_shape=(height_width, height_width, 1))(inputs)
x = layers.Conv2D(filters=32, kernel_size=3, strides=2, activation="relu")(x)
x = layers.Conv2D(filters=64, kernel_size=3, strides=2, activation="relu")(x)
x = layers.Conv2D(filters=128, kernel_size=3, strides=2, activation="relu")(x)
x = layers.GlobalAveragePooling2D()(x)
embeddings = layers.Dense(units=8, activation=None)(x)
embeddings = layers.UnitNormalization()(embeddings)

model = EmbeddingModel(inputs, embeddings)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
)

history = model.fit(AnchorPositivePairs(num_batches=1000), epochs=20)

plt.plot(history.history["loss"])
plt.show()


# 训练的模型效果展示：
near_neighbours_per_example = 10

embeddings = model.predict(x_test)
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
# array([[100.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.],
#        [  0., 100.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.],
#        [  0.,   0., 100.,   0.,   0.,   0.,   0.,   0.,   0.,   0.],
#        [  0.,   0.,   0.,  98.,   0.,   2.,   0.,   0.,   0.,   0.],
#        [  0.,   0.,   0.,   0., 100.,   0.,   0.,   0.,   0.,   0.],
#        [  0.,   0.,   0.,   0.,   0.,  99.,   1.,   0.,   0.,   0.],
#        [  0.,   0.,   0.,   0.,   0.,   0., 100.,   0.,   0.,   0.],
#        [  0.,   0.,   0.,   0.,   0.,   0.,   0., 100.,   0.,   0.],
#        [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0., 100.,   0.],
#        [  0.,   0.,   0.,   0.,   0.,  10.,   0.,   0.,   0.,  90.]])

num = 0
for y_test_idx in range(near_neighbours.shape[0]):
    if len([1 for nn_idx in near_neighbours[y_test_idx][:-1] if y_test[y_test_idx] == y_test[nn_idx]])>5:
        num += 1
print("正确率：", num/near_neighbours.shape[0])
# 正确率： 0.9838

def main():
    pass


if __name__ == "__main__":
    main()
