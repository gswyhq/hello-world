#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Vision Transformers (ViT)
# 来源：https://keras.io/examples/vision/token_learner/

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa

from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

import math


BATCH_SIZE = 256
AUTO = tf.data.AUTOTUNE


# OPTIMIZER
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

# TRAINING
EPOCHS = 20

# AUGMENTATION
IMAGE_SIZE = 48  # We will resize input images to this size.
PATCH_SIZE = 6  # Size of the patches to be extracted from the input images.
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2

# ViT ARCHITECTURE
LAYER_NORM_EPS = 1e-6
PROJECTION_DIM = 128
NUM_HEADS = 4
NUM_LAYERS = 4
MLP_UNITS = [
    PROJECTION_DIM * 2,
    PROJECTION_DIM,
]

# TOKENLEARNER
NUM_TOKENS = 4

# Load the CIFAR-10 dataset.
# (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
# (x_train, y_train), (x_val, y_val) = (
#     (x_train[:40000], y_train[:40000]),
#     (x_train[40000:], y_train[40000:]),
# )

# (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
(x_train, y_train) = (np.random.random(size=(2000, 32, 32, 3)), np.random.randint(10, size=(2000, 1)))
(x_test, y_test) = (np.random.random(size=(500, 32, 32, 3)), np.random.randint(10, size=(500, 1)))
(x_val, y_val) = (np.random.random(size=(500, 32, 32, 3)), np.random.randint(10, size=(500, 1)))
NUM_CLASSES = len(set(y_train[:,0]))
INPUT_SHAPE = x_train.shape[1:]

print(f"Training samples: {len(x_train)}")
print(f"Validation samples: {len(x_val)}")
print(f"Testing samples: {len(x_test)}")

# Convert to tf.data.Dataset objects.
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_ds = train_ds.shuffle(BATCH_SIZE * 100).batch(BATCH_SIZE).prefetch(AUTO)

val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_ds = val_ds.batch(BATCH_SIZE).prefetch(AUTO)

test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_ds = test_ds.batch(BATCH_SIZE).prefetch(AUTO)
# Training samples: 40000
# Validation samples: 10000
# Testing samples: 10000
#

data_augmentation = keras.Sequential(
    [
        layers.Rescaling(1 / 255.0),
        layers.Resizing(INPUT_SHAPE[0] + 20, INPUT_SHAPE[0] + 20),
        layers.RandomCrop(IMAGE_SIZE, IMAGE_SIZE),
        layers.RandomFlip("horizontal"),
    ],
    name="data_augmentation",
)

#位置嵌入模块
#Transformer架构由多个头部自我关注层和作为主要组件的完全连接的前馈网络（MLP）组成。这两个组件都是排列不变的：它们不知道特征顺序。
#为了解决这个问题，我们将位置信息注入令牌。position_embedding函数将此位置信息添加到线性投影标记中。

def position_embedding(
    projected_patches, num_patches=NUM_PATCHES, projection_dim=PROJECTION_DIM
):
    # Build the positions.
    positions = tf.range(start=0, limit=num_patches, delta=1)

    # Encode the positions with an Embedding layer.
    encoded_positions = layers.Embedding(
        input_dim=num_patches, output_dim=projection_dim
    )(positions)

    # Add encoded positions to the projected patches.
    return projected_patches + encoded_positions


# MLP block for Transformer
def mlp(x, dropout_rate, hidden_units):
    # Iterate over the hidden units and
    # add Dense => Dropout.
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

# TokenLearner module
#TokenLearner模块将图像形状的张量作为输入。
# 然后，它通过多个单通道卷积层提取聚焦于输入的不同部分的不同空间注意力图。然后将这些注意力图按元素与输入相乘，并将结果与池进行聚合。
# 这个汇集的输出可以作为输入的摘要，并且具有比原始补丁（例如196个）少得多的补丁（例如8个）。
# 使用多个卷积层有助于表现力。施加某种形式的空间注意力有助于从输入中保留相关信息。这两个组件对于使TokenLearner工作至关重要，尤其是当我们大幅减少补丁数量时。

def token_learner(inputs, number_of_tokens=NUM_TOKENS):
    # Layer normalize the inputs.
    x = layers.LayerNormalization(epsilon=LAYER_NORM_EPS)(inputs)  # (B, H, W, C)

    # Applying Conv2D => Reshape => Permute
    # The reshape and permute is done to help with the next steps of
    # multiplication and Global Average Pooling.
    attention_maps = keras.Sequential(
        [
            # 3 layers of conv with gelu activation as suggested
            # in the paper.
            layers.Conv2D(
                filters=number_of_tokens,
                kernel_size=(3, 3),
                activation=tf.nn.gelu,
                padding="same",
                use_bias=False,
            ),
            layers.Conv2D(
                filters=number_of_tokens,
                kernel_size=(3, 3),
                activation=tf.nn.gelu,
                padding="same",
                use_bias=False,
            ),
            layers.Conv2D(
                filters=number_of_tokens,
                kernel_size=(3, 3),
                activation=tf.nn.gelu,
                padding="same",
                use_bias=False,
            ),
            # This conv layer will generate the attention maps
            layers.Conv2D(
                filters=number_of_tokens,
                kernel_size=(3, 3),
                activation="sigmoid",  # Note sigmoid for [0, 1] output
                padding="same",
                use_bias=False,
            ),
            # Reshape and Permute
            layers.Reshape((-1, number_of_tokens)),  # (B, H*W, num_of_tokens)
            layers.Permute((2, 1)),
        ]
    )(
        x
    )  # (B, num_of_tokens, H*W)

    # Reshape the input to align it with the output of the conv block.
    num_filters = inputs.shape[-1]
    inputs = layers.Reshape((1, -1, num_filters))(inputs)  # inputs == (B, 1, H*W, C)

    # Element-Wise multiplication of the attention maps and the inputs
    attended_inputs = (
        attention_maps[..., tf.newaxis] * inputs
    )  # (B, num_tokens, H*W, C)

    # Global average pooling the element wise multiplication result.
    outputs = tf.reduce_mean(attended_inputs, axis=2)  # (B, num_tokens, C)
    return outputs


# Transformer block
def transformer(encoded_patches):
    # Layer normalization 1.
    x1 = layers.LayerNormalization(epsilon=LAYER_NORM_EPS)(encoded_patches)

    # Multi Head Self Attention layer 1.
    attention_output = layers.MultiHeadAttention(
        num_heads=NUM_HEADS, key_dim=PROJECTION_DIM, dropout=0.1
    )(x1, x1)

    # Skip connection 1.
    x2 = layers.Add()([attention_output, encoded_patches])

    # Layer normalization 2.
    x3 = layers.LayerNormalization(epsilon=LAYER_NORM_EPS)(x2)

    # MLP layer 1.
    x4 = mlp(x3, hidden_units=MLP_UNITS, dropout_rate=0.1)

    # Skip connection 2.
    encoded_patches = layers.Add()([x4, x2])
    return encoded_patches

# ViT model with the TokenLearner module
def create_vit_classifier(use_token_learner=True, token_learner_units=NUM_TOKENS):
    inputs = layers.Input(shape=INPUT_SHAPE)  # (B, H, W, C)

    # Augment data.
    augmented = data_augmentation(inputs)

    # Create patches and project the pathces.
    projected_patches = layers.Conv2D(
        filters=PROJECTION_DIM,
        kernel_size=(PATCH_SIZE, PATCH_SIZE),
        strides=(PATCH_SIZE, PATCH_SIZE),
        padding="VALID",
    )(augmented)
    _, h, w, c = projected_patches.shape
    projected_patches = layers.Reshape((h * w, c))(
        projected_patches
    )  # (B, number_patches, projection_dim)

    # Add positional embeddings to the projected patches.
    encoded_patches = position_embedding(
        projected_patches
    )  # (B, number_patches, projection_dim)
    encoded_patches = layers.Dropout(0.1)(encoded_patches)

    # Iterate over the number of layers and stack up blocks of
    # Transformer.
    for i in range(NUM_LAYERS):
        # Add a Transformer block.
        encoded_patches = transformer(encoded_patches)

        # Add TokenLearner layer in the middle of the
        # architecture. The paper suggests that anywhere
        # between 1/2 or 3/4 will work well.
        if use_token_learner and i == NUM_LAYERS // 2:
            _, hh, c = encoded_patches.shape
            h = int(math.sqrt(hh))
            encoded_patches = layers.Reshape((h, h, c))(
                encoded_patches
            )  # (B, h, h, projection_dim)
            encoded_patches = token_learner(
                encoded_patches, token_learner_units
            )  # (B, num_tokens, c)

    # Layer normalization and Global average pooling.
    representation = layers.LayerNormalization(epsilon=LAYER_NORM_EPS)(encoded_patches)
    representation = layers.GlobalAvgPool1D()(representation)

    # Classify outputs.
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(representation)

    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

# Train and evaluate a ViT with TokenLearner
vit_token_learner = create_vit_classifier()
model = vit_token_learner
# Initialize the AdamW optimizer.
optimizer = tfa.optimizers.AdamW(
    learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY
)

# Compile the model with the optimizer, loss function
# and the metrics.
model.compile(
    optimizer=optimizer,
    loss="sparse_categorical_crossentropy",
    metrics=[
        keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
        keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
    ],
)


# Train the model.
history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds,
)

# Epoch 1/20
#
# 2021-12-15 13:59:59.531011: I tensorflow/stream_executor/cuda/cuda_dnn.cc:366] Loaded cuDNN version 8200
# 2021-12-15 14:00:04.728435: I tensorflow/stream_executor/cuda/cuda_blas.cc:1774] TensorFloat-32 will be used for the matrix multiplication. This will only be logged once.
#
# 157/157 [==============================] - 20s 39ms/step - loss: 2.2716 - accuracy: 0.1396 - top-5-accuracy: 0.5908 - val_loss: 2.0672 - val_accuracy: 0.2004 - val_top-5-accuracy: 0.7632
# Epoch 2/20
# 157/157 [==============================] - 5s 34ms/step - loss: 1.9780 - accuracy: 0.2488 - top-5-accuracy: 0.7917 - val_loss: 1.8621 - val_accuracy: 0.2986 - val_top-5-accuracy: 0.8391
# Epoch 3/20
# 157/157 [==============================] - 5s 34ms/step - loss: 1.8168 - accuracy: 0.3138 - top-5-accuracy: 0.8437 - val_loss: 1.7044 - val_accuracy: 0.3680 - val_top-5-accuracy: 0.8793
# Epoch 4/20
# 157/157 [==============================] - 5s 34ms/step - loss: 1.6765 - accuracy: 0.3701 - top-5-accuracy: 0.8820 - val_loss: 1.6490 - val_accuracy: 0.3857 - val_top-5-accuracy: 0.8809
# Epoch 5/20
# 157/157 [==============================] - 5s 34ms/step - loss: 1.6091 - accuracy: 0.4058 - top-5-accuracy: 0.8978 - val_loss: 1.5899 - val_accuracy: 0.4221 - val_top-5-accuracy: 0.8989
# Epoch 6/20
# 157/157 [==============================] - 5s 34ms/step - loss: 1.5386 - accuracy: 0.4340 - top-5-accuracy: 0.9097 - val_loss: 1.5434 - val_accuracy: 0.4321 - val_top-5-accuracy: 0.9098
# Epoch 7/20
# 157/157 [==============================] - 5s 34ms/step - loss: 1.4944 - accuracy: 0.4481 - top-5-accuracy: 0.9171 - val_loss: 1.4914 - val_accuracy: 0.4674 - val_top-5-accuracy: 0.9146
# Epoch 8/20
# 157/157 [==============================] - 5s 33ms/step - loss: 1.4767 - accuracy: 0.4586 - top-5-accuracy: 0.9179 - val_loss: 1.5280 - val_accuracy: 0.4528 - val_top-5-accuracy: 0.9090
# Epoch 9/20
# 157/157 [==============================] - 5s 34ms/step - loss: 1.4331 - accuracy: 0.4751 - top-5-accuracy: 0.9248 - val_loss: 1.3996 - val_accuracy: 0.4857 - val_top-5-accuracy: 0.9298
# Epoch 10/20
# 157/157 [==============================] - 5s 34ms/step - loss: 1.3990 - accuracy: 0.4925 - top-5-accuracy: 0.9291 - val_loss: 1.3888 - val_accuracy: 0.4872 - val_top-5-accuracy: 0.9308
# Epoch 11/20
# 157/157 [==============================] - 5s 33ms/step - loss: 1.3646 - accuracy: 0.5019 - top-5-accuracy: 0.9355 - val_loss: 1.4330 - val_accuracy: 0.4811 - val_top-5-accuracy: 0.9208
# Epoch 12/20
# 157/157 [==============================] - 5s 34ms/step - loss: 1.3607 - accuracy: 0.5037 - top-5-accuracy: 0.9354 - val_loss: 1.3242 - val_accuracy: 0.5149 - val_top-5-accuracy: 0.9415
# Epoch 13/20
# 157/157 [==============================] - 5s 34ms/step - loss: 1.3303 - accuracy: 0.5170 - top-5-accuracy: 0.9384 - val_loss: 1.2934 - val_accuracy: 0.5295 - val_top-5-accuracy: 0.9437
# Epoch 14/20
# 157/157 [==============================] - 5s 33ms/step - loss: 1.3038 - accuracy: 0.5259 - top-5-accuracy: 0.9426 - val_loss: 1.3102 - val_accuracy: 0.5187 - val_top-5-accuracy: 0.9422
# Epoch 15/20
# 157/157 [==============================] - 5s 33ms/step - loss: 1.2926 - accuracy: 0.5304 - top-5-accuracy: 0.9441 - val_loss: 1.3220 - val_accuracy: 0.5234 - val_top-5-accuracy: 0.9428
# Epoch 16/20
# 157/157 [==============================] - 5s 34ms/step - loss: 1.2724 - accuracy: 0.5346 - top-5-accuracy: 0.9458 - val_loss: 1.2670 - val_accuracy: 0.5370 - val_top-5-accuracy: 0.9491
# Epoch 17/20
# 157/157 [==============================] - 5s 33ms/step - loss: 1.2515 - accuracy: 0.5450 - top-5-accuracy: 0.9462 - val_loss: 1.2837 - val_accuracy: 0.5349 - val_top-5-accuracy: 0.9474
# Epoch 18/20
# 157/157 [==============================] - 5s 33ms/step - loss: 1.2427 - accuracy: 0.5505 - top-5-accuracy: 0.9492 - val_loss: 1.3425 - val_accuracy: 0.5180 - val_top-5-accuracy: 0.9371
# Epoch 19/20
# 157/157 [==============================] - 5s 34ms/step - loss: 1.2129 - accuracy: 0.5605 - top-5-accuracy: 0.9514 - val_loss: 1.2297 - val_accuracy: 0.5590 - val_top-5-accuracy: 0.9536
# Epoch 20/20
# 157/157 [==============================] - 5s 33ms/step - loss: 1.1994 - accuracy: 0.5667 - top-5-accuracy: 0.9523 - val_loss: 1.2390 - val_accuracy: 0.5577 - val_top-5-accuracy: 0.9528
# 40/40 [==============================] - 0s 11ms/step - loss: 1.2293 - accuracy: 0.5564 - top-5-accuracy: 0.9549
# Test accuracy: 55.64%
# Test top 5 accuracy: 95.49%

def main():
    pass


if __name__ == '__main__':
    main()
