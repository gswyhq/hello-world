#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 变分自编码器（Variational Autoencoder, VAE）
# 需要一个Encoder编码器类，它把一个MNIST手写数字图片来对应到一个在潜在空间（latent space）里面的三元组(z_mean, z_log_var, z)，过程中使用了一个Sampling采样层。
# 下一步，我们用一个Decoder解码器类来把潜在空间的坐标对应回到一个MNIST数字图片。
# 最后，我们的VariationalAutoEncoder变分自编码器类会把编码器和解码器串起来，然后用add_loss()来加入KL散度正则化的损失函数。

# 来源：https://zhuanlan.zhihu.com/p/380472423

import tensorflow as tf
from tensorflow import keras

original_dim = 784
intermediate_dim = 64
latent_dim = 32

# 编码器
original_inputs = tf.keras.Input(shape=(original_dim,), name="encoder_input")
x = layers.Dense(intermediate_dim, activation="relu")(original_inputs)
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
z = Sampling()((z_mean, z_log_var))
encoder = tf.keras.Model(inputs=original_inputs, outputs=z, name="encoder")

# 解码器
latent_inputs = tf.keras.Input(shape=(latent_dim,), name="z_sampling")
x = layers.Dense(intermediate_dim, activation="relu")(latent_inputs)
outputs = layers.Dense(original_dim, activation="sigmoid")(x)
decoder = tf.keras.Model(inputs=latent_inputs, outputs=outputs, name="decoder")

# VAE模型
outputs = decoder(z)
vae = tf.keras.Model(inputs=original_inputs, outputs=outputs, name="vae")

# 添加KL散度正则化损失函数
kl_loss = -0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
vae.add_loss(kl_loss)

# 损失函数和优化器
loss_fn = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

# 准备数据
(x_train, _), _ = tf.keras.datasets.mnist.load_data()
dataset = tf.data.Dataset.from_tensor_slices(
    x_train.reshape(60000, 784).astype("float32") / 255
)
dataset = dataset.map(lambda x: (x, x))  # 用x_train同时作为输入和输出目标
dataset = dataset.shuffle(buffer_size=1024).batch(32)

# 配置模型，准备训练
vae.compile(optimizer, loss=loss_fn)

# 对模型进行训练
vae.fit(dataset, epochs=1)

def main():
    pass


if __name__ == '__main__':
    main()
