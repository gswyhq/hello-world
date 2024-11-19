#!/usr/bin/env python
# coding=utf-8

# 来源：https://keras.io/examples/vision/siamese_network/

import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import random
import collections
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

from pathlib import Path
from tensorflow.keras import applications
from tensorflow.keras import layers
from tensorflow.keras import losses
# from tensorflow.keras import ops
from tensorflow.keras import optimizers
from tensorflow.keras import metrics
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.applications import resnet
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

target_shape = (28, 28)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1666666)

x_train = x_train.astype("float32") / 255.0
y_train = np.squeeze(y_train)
x_val = x_val.astype("float32") / 255.0
y_val = np.squeeze(y_val)
x_test = x_test.astype("float32") / 255.0
y_test = np.squeeze(y_test)

class_idx_to_train_idxs = defaultdict(list)
for y_train_idx, y in enumerate(y_train):
    class_idx_to_train_idxs[y].append(y_train_idx)

class_idx_to_val_idxs = defaultdict(list)
for y_val_idx, y in enumerate(y_val):
    class_idx_to_val_idxs[y].append(y_val_idx)
    
class_idx_to_test_idxs = defaultdict(list)
for y_test_idx, y in enumerate(y_test):
    class_idx_to_test_idxs[y].append(y_test_idx)

num_classes = 10
height_width = 28

# anchor是基准
# positive是针对anchor的正样本，表示与anchor来自同一个人
# negative是针对anchor的负样本
###########################################################################################################################
# 第一步，先简单训练个分类模型，先看看正确率能达到多少，最后达到了99%，说明用于训练的样本过于简单，对比学习可能发挥不了作用；
keras.backend.clear_session()
input = layers.Input(shape=target_shape+(1,))
x = layers.Conv2D(32, kernel_size=(3, 3), activation="relu")(input)
x = layers.MaxPooling2D(pool_size=(2, 2))(x)
x = layers.Conv2D(64, kernel_size=(3, 3), activation="relu")(x)
x = layers.MaxPooling2D(pool_size=(2, 2))(x)
x = layers.Flatten()(x)
x = layers.Dropout(0.5)(x)
output = layers.Dense(num_classes, activation="softmax", name='output')(x)
cls_model = Model(input, output)
batch_size = 128
epochs = 15

cls_model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
cls_model.summary()
history = cls_model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_val, y_val))
cls_model.save('./results/MNIST_cls_model.hdf5')
print(history.history)
# {'loss': [0.39034250378608704, 0.11696343868970871, 0.08677700161933899, 0.07248110324144363, 0.06118470057845116, 0.055245377123355865, 0.05252356454730034, 0.047678712755441666, 0.044471997767686844, 0.04147888720035553, 0.039106570184230804, 0.03816516324877739, 0.03552449122071266, 0.03244669362902641, 0.03226446360349655], 'accuracy': [0.879580020904541, 0.9639999866485596, 0.972599983215332, 0.9769999980926514, 0.9809200167655945, 0.9830399751663208, 0.9840199947357178, 0.9848799705505371, 0.9857199788093567, 0.9866799712181091, 0.9872999787330627, 0.9879400134086609, 0.9885799884796143, 0.9891600012779236, 0.9898999929428101], 'val_loss': [0.11528768390417099, 0.0817207545042038, 0.060283929109573364, 0.05005499720573425, 0.04793082922697067, 0.04369591549038887, 0.041275132447481155, 0.03964625298976898, 0.0388716496527195, 0.039507970213890076, 0.03585972264409065, 0.03445514291524887, 0.034064531326293945, 0.034035880118608475, 0.03270241618156433], 'val_accuracy': [0.96670001745224, 0.9750000238418579, 0.982200026512146, 0.9848999977111816, 0.9853000044822693, 0.9872000217437744, 0.9866999983787537, 0.9882000088691711, 0.9879000186920166, 0.988099992275238, 0.9894000291824341, 0.9900000095367432, 0.989300012588501, 0.989300012588501, 0.9902999997138977]}

plt.plot(history.history['loss'])
plt.show()

score = cls_model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

# Test loss: 0.026941223070025444
# Test accuracy: 0.9904999732971191

###########################################################################################################################
# 第二步：使用triplet_loss，训练孪生网络；

def generate_anchor_positive_negative_pairs(x_train, y_train, batch_size):

    anchor, positive, negative = [], [], []
    while True:
        x_train, y_train = shuffle(x_train, y_train)
        class_idx_to_train_idxs = defaultdict(list)
        for y_train_idx, y in enumerate(y_train):
            class_idx_to_train_idxs[y].append(y_train_idx)

        for anchor_idx, class_idx in enumerate(y_train):
            examples_for_class = class_idx_to_train_idxs[class_idx]
            # anchor_idx = random.choice(examples_for_class)
            positive_idx = random.choice(examples_for_class)
            while positive_idx == anchor_idx:
                positive_idx = random.choice(examples_for_class)
            diff_class_idx = random.choice([i for i in range(num_classes) if i != class_idx])
            negative_idx = random.choice(class_idx_to_train_idxs[diff_class_idx])
            anchor.append(x_train[anchor_idx])
            positive.append(x_train[positive_idx])
            negative.append(x_train[negative_idx])

            if len(anchor) == batch_size:
                yield {'anchor': np.array(anchor),
                        'positive': np.array(positive),
                        'negative': np.array(negative)}
                anchor, positive, negative = [], [], []

def generate_hard_pairs(x_train, y_train, batch_size, margin=0.5, embedding_model=None):
    '''选择困难样本
    若不选择困难样本，模型可能不收敛；不管margin设置多少 每次都稳定在margin附近，降低学习率也没有办法，这可能是训练样本选择的不对
    dist(x,xn)表示特征向量x与负样本之间的距离。
    dist(x,xp)表示特征向量x与正样本之间的距离。
    semi-hard negatives，此类xn满足
    dist(x,xp) < dist(x,xn) < dist(x,xp)+margin
    正常情况下是：负样本的距离（dist(x,xn)）要大于anchor样本和正样本之间的距离（dist(x,xp)）。这是为了保证负样本和正样本之间的距离足够大，以便于区分。
    但训练模型是要选择预测错误的样本来加强模型训练，即 dist(x, xn) < dist(x, xp), 但又不能小太多，还要求 dist(x, xp) - margin < dist(x, xn) < dist(x, xp)
    '''
    num_hard = max(batch_size//5, 2)
    num_rand = batch_size - num_hard
    for data in generate_anchor_positive_negative_pairs(x_train, y_train, batch_size+num_rand):
        A_emb = embedding_model.predict(data['anchor'][:batch_size], verbose=0)
        P_emb = embedding_model.predict(data['positive'][:batch_size], verbose=0)
        N_emb = embedding_model.predict(data['negative'][:batch_size], verbose=0)

        # Compute d(A, P) - d(A, N) for each selected batch
        batch_losses = np.sum(np.square(A_emb - P_emb), axis=-1) - np.sum(np.square(A_emb - N_emb), axis=-1)

        hard_batch_indices = [x for x, _ in sorted(enumerate(batch_losses), key=lambda x: np.abs(x[1]-margin/2) )]
        hard_batches = hard_batch_indices[:num_hard]
        rand_batches = [i for i in range(batch_size, batch_size+num_rand)]


        selections = {'anchor': data['anchor'][hard_batches + rand_batches],
                        'positive': data['positive'][hard_batches + rand_batches],
                        'negative': data['negative'][hard_batches + rand_batches]
                      }
        yield selections

# 可视化训练样本对
def plot_triplets(examples, x_train_w=28, x_train_h=28):
    plt.figure(figsize=(6, 2))
    for i in range(3):
        plt.subplot(1, 3, 1 + i)
        plt.imshow(np.reshape(examples[i], (x_train_w, x_train_h)), cmap='binary')
        plt.xticks([])
        plt.yticks([])
    plt.show()

batch_size = 512
selections = next(generate_anchor_positive_negative_pairs(x_train, y_train, batch_size))
plot_triplets([selections['anchor'][2], selections['positive'][2], selections['negative'][2]])

########################################################################################################################
# 训练模型，模型可能不收敛；不管margin设置多少 每次都稳定在margin附近，降低学习率也没有办法，这可能是训练样本选择的不对，但不影响本次演示；
# 模型训练完，只要是锚点与正样本的距离小于描点与负样本间的距离就属正常。

keras.backend.clear_session()

inputs = layers.Input(shape=(height_width, height_width))
x = Reshape(target_shape=(height_width, height_width, 1))(inputs)
x = layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu")(x)
x = layers.MaxPooling2D(pool_size=(2, 2))(x)
x = layers.Conv2D(64, kernel_size=(3, 3), activation="relu")(x)
x = layers.MaxPooling2D(pool_size=(2, 2))(x)
x = layers.Flatten()(x)
x = layers.Dropout(0.5)(x)
output = layers.Dense(256)(x)
embedding = Model(inputs, output, name="Embedding")

# 查看选择的困难样本
selections = next(generate_hard_pairs(x_train, y_train, batch_size, margin=0.5, embedding_model=embedding))
plot_triplets([selections['anchor'][2], selections['positive'][2], selections['negative'][2]])

class DistanceLayer(layers.Layer):
    """
    This layer is responsible for computing the distance between the anchor
    embedding and the positive embedding, and the anchor embedding and the
    negative embedding.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, anchor, positive, negative):
        ap_distance = K.sum(tf.square(anchor - positive), -1)
        an_distance = K.sum(tf.square(anchor - negative), -1)
        return (ap_distance, an_distance)


anchor_input = layers.Input(name="anchor", shape=target_shape )
positive_input = layers.Input(name="positive", shape=target_shape )
negative_input = layers.Input(name="negative", shape=target_shape )

distances = DistanceLayer()(
    embedding(anchor_input),
    embedding(positive_input),
    embedding(negative_input),
)

siamese_network = Model(
    inputs=[anchor_input, positive_input, negative_input], outputs=distances
)

class SiameseModel(Model):
    """The Siamese Network model with a custom training and testing loops.

    Computes the triplet loss using the three embeddings produced by the
    Siamese Network.

    The triplet loss is defined as:
       L(A, P, N) = max(‖f(A) - f(P)‖² - ‖f(A) - f(N)‖² + margin, 0)
    """

    def __init__(self, siamese_network, margin=0.5):
        super().__init__()
        self.siamese_network = siamese_network
        self.margin = margin
        self.loss_tracker = metrics.Mean(name="loss")

    def call(self, inputs):
        return self.siamese_network(inputs)

    def train_step(self, data):
        # GradientTape is a context manager that records every operation that
        # you do inside. We are using it here to compute the loss so we can get
        # the gradients and apply them using the optimizer specified in
        # `compile()`.
        with tf.GradientTape() as tape:
            loss = self._compute_loss(data)

        # Storing the gradients of the loss function with respect to the
        # weights/parameters.
        gradients = tape.gradient(loss, self.siamese_network.trainable_weights)

        # Applying the gradients on the model using the specified optimizer
        self.optimizer.apply_gradients(
            zip(gradients, self.siamese_network.trainable_weights)
        )

        # Let's update and return the training loss metric.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        loss = self._compute_loss(data)

        # Let's update and return the loss metric.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def _compute_loss(self, data):
        ap_distance, an_distance = self.siamese_network(data)
        loss = ap_distance - an_distance
        loss = tf.maximum(loss + self.margin, 0.0)
        return loss

    @property
    def metrics(self):
        # We need to list our metrics here so the `reset_states()` can be
        # called automatically.
        return [self.loss_tracker]
    
batch_size = 512
margin=1.0
siamese_model = SiameseModel(siamese_network, margin=margin)
siamese_model.compile(optimizer=optimizers.Adam(5e-5))
history = siamese_model.fit(
                            generate_hard_pairs(x_train, y_train, batch_size, margin=margin, embedding_model=embedding),
                            # generate_anchor_positive_negative_pairs(x_train, y_train, batch_size),
                            steps_per_epoch=x_train.shape[0]//batch_size,
                            epochs=5, 
                            validation_data=generate_hard_pairs(x_val, y_val, batch_size, margin=margin, embedding_model=embedding),
                            # validation_data=generate_anchor_positive_negative_pairs(x_val, y_val, batch_size),
                            validation_steps=x_val.shape[0]//batch_size, 
    )
print(history.history)
# Epoch 1/5
# 97/97 [==============================] - 52s 533ms/step - loss: 0.9890 - val_loss: 0.9456
# Epoch 2/5
# 97/97 [==============================] - 59s 607ms/step - loss: 0.9408 - val_loss: 0.9312
# Epoch 3/5
# 97/97 [==============================] - 59s 616ms/step - loss: 0.9308 - val_loss: 0.9397
# Epoch 4/5
# 97/97 [==============================] - 62s 639ms/step - loss: 0.9298 - val_loss: 0.9303
# Epoch 5/5
# 97/97 [==============================] - 67s 694ms/step - loss: 0.9299 - val_loss: 0.9188
# {'loss': [0.9889981150627136, 0.9408355951309204, 0.930802047252655, 0.9297903776168823, 0.9298543334007263], 'val_loss': [0.9456098675727844, 0.931239902973175, 0.9396767616271973, 0.9302985668182373, 0.9187825918197632]}

plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()


###########################################################################################################################
# 我们可以计算锚点和正样本之间的余弦相似度，并将其与锚点和负样本之间的相似度进行比较。
# 我们应该期望锚点和正样本之间的相似性大于锚点和负样本之间的相似性。

selections = next(generate_anchor_positive_negative_pairs(x_train, y_train, batch_size))

anchor_embedding, positive_embedding, negative_embedding  = embedding(np.array([selections['anchor'][2], selections['positive'][2], selections['negative'][2]]))

cosine_similarity = metrics.CosineSimilarity()

positive_similarity = cosine_similarity(anchor_embedding, positive_embedding)
print("Positive similarity:", positive_similarity.numpy())

negative_similarity = cosine_similarity(anchor_embedding, negative_embedding)
print("Negative similarity", negative_similarity.numpy())

###########################################################################################################################
# 进一步的，我们取训练样本作为标准集，用测试集测试，看看正确率是多少：
y_train_pred = embedding(x_train)
y_test_pred = embedding(x_test)

# 计算测试样本与训练集所有样本的余弦相似度
test_embedding = tf.expand_dims(y_test_pred, axis=1)
train100_embedding = tf.expand_dims(y_train_pred, axis=0)

dot_product = tf.einsum('ijk,jnk->in', test_embedding, train100_embedding)
norm_a = tf.norm(test_embedding, axis=-1)
norm_p = tf.norm(train100_embedding, axis=-1)
similarity = dot_product / (norm_a * norm_p)

# 最相似训练样本的标签即认为是对应的预测标签，并计算预测正确率
print("正确率：", len([k for k, v in zip([y_train[i] for i in similarity.numpy().argmax(axis=-1)], y_test) if k == v])/y_test.shape[0])
# 正确率： 0.9467

# 取最相似的top100,计算正确率
top100 = similarity.numpy().argsort(axis=-1)[:,-100:]
print("正确率2：", len([k for k, v in zip([collections.Counter(y_train[i]).most_common(1)[0][0] for i in top100], y_test) if k == v])/y_test.shape[0])
# 正确率2： 0.861

# 这个正确率，对比上面的99%正确率，效果还是很差的，原因可能是样本没有经过精心选择等原因；导致训练的所有向量趋向于0


def main():
    pass


if __name__ == "__main__":
    main()
