#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 来源： https://keras.io/examples/vision/cct/
# Compact Convolutional Transformer (CCT)
from tensorflow.keras import layers
from tensorflow import keras

import matplotlib.pyplot as plt
import tensorflow_addons as tfa
import tensorflow as tf
import numpy as np


positional_emb = True
conv_layers = 2
projection_dim = 128

num_heads = 2
transformer_units = [
    projection_dim,
    projection_dim,
]
transformer_layers = 2
stochastic_depth_rate = 0.1

learning_rate = 0.001
weight_decay = 0.0001
batch_size = 128
num_epochs = 30
image_size = 32

# (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
(x_train, y_train), (x_test, y_test) = (np.random.random(size=(2000, 32, 32, 3)), np.random.randint(10, size=(2000, 1))), (np.random.random(size=(500, 32, 32, 3)), np.random.randint(10, size=(500, 1)))

num_classes = len(set(y_train[:,0]))
input_shape = x_train.shape[1:]

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

print(f"x_train shape: {x_train.shape} - y_train shape: {y_train.shape}")
print(f"x_test shape: {x_test.shape} - y_test shape: {y_test.shape}")
# x_train shape: (50000, 32, 32, 3) - y_train shape: (50000, 10)
# x_test shape: (10000, 32, 32, 3) - y_test shape: (10000, 10)

class CCTTokenizer(layers.Layer):
    def __init__(
        self,
        kernel_size=3,
        stride=1,
        padding=1,
        pooling_kernel_size=3,
        pooling_stride=2,
        num_conv_layers=conv_layers,
        num_output_channels=[64, 128],
        positional_emb=positional_emb,
        **kwargs,
    ):
        super(CCTTokenizer, self).__init__(**kwargs)

        # This is our tokenizer.
        self.conv_model = keras.Sequential()
        for i in range(num_conv_layers):
            self.conv_model.add(
                layers.Conv2D(
                    num_output_channels[i],
                    kernel_size,
                    stride,
                    padding="valid",
                    use_bias=False,
                    activation="relu",
                    kernel_initializer="he_normal",
                )
            )
            self.conv_model.add(layers.ZeroPadding2D(padding))
            self.conv_model.add(
                layers.MaxPool2D(pooling_kernel_size, pooling_stride, "same")
            )

        self.positional_emb = positional_emb

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            "conv_model": self.conv_model,
            "positional_emb": self.positional_emb,
        })
        return config

    def call(self, images):
        outputs = self.conv_model(images) # -> shape=(None, 8, 8, 128)
        # After passing the images through our mini-network the spatial dimensions
        # are flattened to form sequences.
        reshaped = tf.reshape(
            outputs,
            (-1, tf.shape(outputs)[1] * tf.shape(outputs)[2], tf.shape(outputs)[-1]),
        ) # -> shape=(None, 8*8, 128)
        return reshaped

    def positional_embedding(self, image_size):
        # Positional embeddings are optional in CCT. Here, we calculate
        # the number of sequences and initialize an `Embedding` layer to
        # compute the positional embeddings later.
        if self.positional_emb:
            dummy_inputs = tf.ones((1, image_size, image_size, 3))
            dummy_outputs = self.call(dummy_inputs)  # -> shape=(None, 8*8, 128)
            sequence_length = tf.shape(dummy_outputs)[1]  # -> 64
            projection_dim = tf.shape(dummy_outputs)[-1]  # -> 128

            embed_layer = layers.Embedding(
                input_dim=sequence_length, output_dim=projection_dim
            )
            return embed_layer, sequence_length
        else:
            return None

# 随机深度正则化(Stochastic depth for regularization)
# 通过增加卷积网络的深度能显著降低预测误差，提高网络的表达能力，然而随着网络层的加深，也会带来一些负面影响，比如梯度消失，前向传播耗时增加、训练缓慢、模型过拟合训练数据等等。为了解决这些问题，我们提出了随机深度的方法，一个看似矛盾的设置，在训练过程降低网络的深度，在测试阶段保持网络的深度。通过该方法，大大减少了训练时间，并在评估的一些数据集上显著改善了测试误差。

# Referred from: github.com:rwightman/pytorch-image-models.
class StochasticDepth(layers.Layer):
    def __init__(self, drop_prop, **kwargs):
        super(StochasticDepth, self).__init__(**kwargs)
        self.drop_prob = drop_prop

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            "drop_prob": self.drop_prob,
        })
        return config

    def call(self, x, training=None):
        if training:
            keep_prob = 1 - self.drop_prob
            shape = (tf.shape(x)[0],) + (1,) * (tf.shape(x).shape[0] - 1)
            random_tensor = keep_prob + tf.random.uniform(shape, 0, 1)
            random_tensor = tf.floor(random_tensor)
            return (x / keep_prob) * random_tensor
        return x

# MLP for the Transformers encoder
def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

# 数据增强(Data augmentation)
# 在最初的论文中，作者使用AutoAugment来诱导更强的正则化。在本例中，我们将使用标准的几何扩充，如随机裁剪和翻转。

# 注意重新缩放图层。这些层具有预定义的推理行为。
data_augmentation = keras.Sequential(
    [
        layers.Rescaling(scale=1.0 / 255),
        layers.RandomCrop(image_size, image_size),
        layers.RandomFlip("horizontal"),
    ],
    name="data_augmentation",
)


# 最终CCT模型
# CCT中引入的另一个配方是注意力集中或序列集中。在ViT中，只有与类令牌对应的特征图被合并，然后用于后续的分类任务（或任何其他下游任务）。
# 在CCT中，Transformers编码器的输出被加权，然后传递到最终的任务特定层（在本例中，我们进行分类）。

def create_cct_model(
    image_size=image_size,
    input_shape=input_shape,
    num_heads=num_heads,
    projection_dim=projection_dim,
    transformer_units=transformer_units,
):

    inputs = layers.Input(input_shape) # -> shape=(None, 32, 32, 3)

    # Augment data.
    augmented = data_augmentation(inputs) # -> shape=(None, 32, 32, 3)

    # Encode patches.
    cct_tokenizer = CCTTokenizer()
    encoded_patches = cct_tokenizer(augmented) # -> shape=(None, 64, 128)

    # Apply positional embedding.
    if positional_emb:
        pos_embed, seq_length = cct_tokenizer.positional_embedding(image_size)
        positions = tf.range(start=0, limit=seq_length, delta=1)
        position_embeddings = pos_embed(positions)
        encoded_patches += position_embeddings

    # Calculate Stochastic Depth probabilities.
    dpr = [x for x in np.linspace(0, stochastic_depth_rate, transformer_layers)]

    # Create multiple layers of the Transformer block.
    for i in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-5)(encoded_patches)

        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)

        # Skip connection 1.
        attention_output = StochasticDepth(dpr[i])(attention_output)
        x2 = layers.Add()([attention_output, encoded_patches])

        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-5)(x2)

        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)

        # Skip connection 2.
        x3 = StochasticDepth(dpr[i])(x3)
        encoded_patches = layers.Add()([x3, x2])

    # Apply sequence pooling.
    representation = layers.LayerNormalization(epsilon=1e-5)(encoded_patches)
    attention_weights = tf.nn.softmax(layers.Dense(1)(representation), axis=1)
    weighted_representation = tf.matmul(
        attention_weights, representation, transpose_a=True
    )
    weighted_representation = tf.squeeze(weighted_representation, -2)

    # Classify outputs.
    logits = layers.Dense(num_classes)(weighted_representation)
    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=logits)
    return model

# 模型训练集评估
def run_experiment(model):
    optimizer = tfa.optimizers.AdamW(learning_rate=0.001, weight_decay=0.0001)

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.CategoricalCrossentropy(
            from_logits=True, label_smoothing=0.1
        ),
        metrics=[
            keras.metrics.CategoricalAccuracy(name="accuracy"),
            keras.metrics.TopKCategoricalAccuracy(5, name="top-5-accuracy"),
        ],
    )

    checkpoint_filepath = "/tmp/checkpoint"
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
    )

    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=batch_size,
        epochs=num_epochs,
        validation_split=0.1,
        callbacks=[checkpoint_callback],
    )

    model.load_weights(checkpoint_filepath)
    _, accuracy, top_5_accuracy = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")

    return history


cct_model = create_cct_model()
history = run_experiment(cct_model)

# Epoch 1/30
# 352/352 [==============================] - 16s 37ms/step - loss: 1.9286 - accuracy: 0.3262 - top-5-accuracy: 0.8222 - val_loss: 1.6803 - val_accuracy: 0.4624 - val_top-5-accuracy: 0.9074
# Epoch 2/30
# 352/352 [==============================] - 12s 35ms/step - loss: 1.5919 - accuracy: 0.4884 - top-5-accuracy: 0.9280 - val_loss: 1.5446 - val_accuracy: 0.5176 - val_top-5-accuracy: 0.9404
# Epoch 3/30
# 352/352 [==============================] - 12s 35ms/step - loss: 1.4632 - accuracy: 0.5535 - top-5-accuracy: 0.9492 - val_loss: 1.3702 - val_accuracy: 0.6046 - val_top-5-accuracy: 0.9574
# Epoch 4/30
# 352/352 [==============================] - 12s 35ms/step - loss: 1.3749 - accuracy: 0.5965 - top-5-accuracy: 0.9588 - val_loss: 1.2989 - val_accuracy: 0.6378 - val_top-5-accuracy: 0.9696
# Epoch 5/30
# 352/352 [==============================] - 12s 35ms/step - loss: 1.3095 - accuracy: 0.6282 - top-5-accuracy: 0.9651 - val_loss: 1.3252 - val_accuracy: 0.6280 - val_top-5-accuracy: 0.9668
# Epoch 6/30
# 352/352 [==============================] - 12s 35ms/step - loss: 1.2735 - accuracy: 0.6483 - top-5-accuracy: 0.9687 - val_loss: 1.2445 - val_accuracy: 0.6658 - val_top-5-accuracy: 0.9750
# Epoch 7/30
# 352/352 [==============================] - 12s 35ms/step - loss: 1.2405 - accuracy: 0.6623 - top-5-accuracy: 0.9712 - val_loss: 1.2127 - val_accuracy: 0.6800 - val_top-5-accuracy: 0.9734
# Epoch 8/30
# 352/352 [==============================] - 13s 36ms/step - loss: 1.1953 - accuracy: 0.6852 - top-5-accuracy: 0.9760 - val_loss: 1.1579 - val_accuracy: 0.7042 - val_top-5-accuracy: 0.9764
# Epoch 9/30
# 352/352 [==============================] - 12s 35ms/step - loss: 1.1659 - accuracy: 0.6940 - top-5-accuracy: 0.9787 - val_loss: 1.1817 - val_accuracy: 0.7026 - val_top-5-accuracy: 0.9746
# Epoch 10/30
# 352/352 [==============================] - 12s 35ms/step - loss: 1.1469 - accuracy: 0.7097 - top-5-accuracy: 0.9784 - val_loss: 1.2331 - val_accuracy: 0.6684 - val_top-5-accuracy: 0.9758
# Epoch 11/30
# 352/352 [==============================] - 12s 35ms/step - loss: 1.1214 - accuracy: 0.7196 - top-5-accuracy: 0.9800 - val_loss: 1.1374 - val_accuracy: 0.7222 - val_top-5-accuracy: 0.9796
# Epoch 12/30
# 352/352 [==============================] - 13s 36ms/step - loss: 1.1055 - accuracy: 0.7264 - top-5-accuracy: 0.9818 - val_loss: 1.1257 - val_accuracy: 0.7276 - val_top-5-accuracy: 0.9796
# Epoch 13/30
# 352/352 [==============================] - 12s 35ms/step - loss: 1.0904 - accuracy: 0.7337 - top-5-accuracy: 0.9820 - val_loss: 1.1029 - val_accuracy: 0.7374 - val_top-5-accuracy: 0.9794
# Epoch 14/30
# 352/352 [==============================] - 12s 35ms/step - loss: 1.0629 - accuracy: 0.7483 - top-5-accuracy: 0.9842 - val_loss: 1.1196 - val_accuracy: 0.7260 - val_top-5-accuracy: 0.9792
# Epoch 15/30
# 352/352 [==============================] - 12s 35ms/step - loss: 1.0558 - accuracy: 0.7528 - top-5-accuracy: 0.9837 - val_loss: 1.1100 - val_accuracy: 0.7308 - val_top-5-accuracy: 0.9780
# Epoch 16/30
# 352/352 [==============================] - 12s 35ms/step - loss: 1.0440 - accuracy: 0.7567 - top-5-accuracy: 0.9850 - val_loss: 1.0782 - val_accuracy: 0.7454 - val_top-5-accuracy: 0.9830
# Epoch 17/30
# 352/352 [==============================] - 12s 35ms/step - loss: 1.0327 - accuracy: 0.7607 - top-5-accuracy: 0.9861 - val_loss: 1.0865 - val_accuracy: 0.7418 - val_top-5-accuracy: 0.9824
# Epoch 18/30
# 352/352 [==============================] - 12s 35ms/step - loss: 1.0160 - accuracy: 0.7695 - top-5-accuracy: 0.9870 - val_loss: 1.0525 - val_accuracy: 0.7594 - val_top-5-accuracy: 0.9822
# Epoch 19/30
# 352/352 [==============================] - 12s 35ms/step - loss: 1.0099 - accuracy: 0.7738 - top-5-accuracy: 0.9867 - val_loss: 1.0568 - val_accuracy: 0.7512 - val_top-5-accuracy: 0.9830
# Epoch 20/30
# 352/352 [==============================] - 12s 35ms/step - loss: 0.9964 - accuracy: 0.7798 - top-5-accuracy: 0.9880 - val_loss: 1.0645 - val_accuracy: 0.7542 - val_top-5-accuracy: 0.9804
# Epoch 21/30
# 352/352 [==============================] - 12s 35ms/step - loss: 0.9929 - accuracy: 0.7807 - top-5-accuracy: 0.9880 - val_loss: 1.0358 - val_accuracy: 0.7692 - val_top-5-accuracy: 0.9832
# Epoch 22/30
# 352/352 [==============================] - 12s 35ms/step - loss: 0.9796 - accuracy: 0.7854 - top-5-accuracy: 0.9889 - val_loss: 1.0191 - val_accuracy: 0.7748 - val_top-5-accuracy: 0.9844
# Epoch 23/30
# 352/352 [==============================] - 12s 35ms/step - loss: 0.9779 - accuracy: 0.7882 - top-5-accuracy: 0.9879 - val_loss: 1.0452 - val_accuracy: 0.7654 - val_top-5-accuracy: 0.9810
# Epoch 24/30
# 352/352 [==============================] - 12s 35ms/step - loss: 0.9728 - accuracy: 0.7901 - top-5-accuracy: 0.9889 - val_loss: 1.0324 - val_accuracy: 0.7674 - val_top-5-accuracy: 0.9822
# Epoch 25/30
# 352/352 [==============================] - 12s 35ms/step - loss: 0.9630 - accuracy: 0.7948 - top-5-accuracy: 0.9885 - val_loss: 1.0611 - val_accuracy: 0.7620 - val_top-5-accuracy: 0.9844
# Epoch 26/30
# 352/352 [==============================] - 12s 35ms/step - loss: 0.9569 - accuracy: 0.7965 - top-5-accuracy: 0.9902 - val_loss: 1.0451 - val_accuracy: 0.7700 - val_top-5-accuracy: 0.9840
# Epoch 27/30
# 352/352 [==============================] - 12s 35ms/step - loss: 0.9466 - accuracy: 0.8030 - top-5-accuracy: 0.9901 - val_loss: 1.0123 - val_accuracy: 0.7824 - val_top-5-accuracy: 0.9874
# Epoch 28/30
# 352/352 [==============================] - 12s 35ms/step - loss: 0.9402 - accuracy: 0.8054 - top-5-accuracy: 0.9902 - val_loss: 0.9999 - val_accuracy: 0.7784 - val_top-5-accuracy: 0.9858
# Epoch 29/30
# 352/352 [==============================] - 12s 35ms/step - loss: 0.9365 - accuracy: 0.8070 - top-5-accuracy: 0.9905 - val_loss: 0.9993 - val_accuracy: 0.7866 - val_top-5-accuracy: 0.9850
# Epoch 30/30
# 352/352 [==============================] - 12s 35ms/step - loss: 0.9373 - accuracy: 0.8045 - top-5-accuracy: 0.9906 - val_loss: 1.0009 - val_accuracy: 0.7870 - val_top-5-accuracy: 0.9864
# 313/313 [==============================] - 2s 7ms/step - loss: 1.0088 - accuracy: 0.7761 - top-5-accuracy: 0.9844
# Test accuracy: 77.61%
# Test top 5 accuracy: 98.44%

# 现在让我们可视化模型的训练进度。
plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Train and Validation Losses Over Epochs", fontsize=14)
plt.legend()
plt.grid()
plt.show()

def main():
    pass


if __name__ == '__main__':
    main()
