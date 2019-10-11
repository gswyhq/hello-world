#!/usr/bin/python3
# coding: utf-8


import os,random
os.environ["KERAS_BACKEND"] = "tensorflow"
import numpy as np
import theano as th
import theano.tensor as T
from keras.utils import np_utils
import keras.models as models
from keras.layers import Input,merge
from keras.layers.core import Reshape,Dense,Dropout,Activation,Flatten,MaxoutDense
from keras.layers.advanced_activations import LeakyReLU
from keras.activations import *
from keras.layers.wrappers import TimeDistributed
from keras.layers.noise import GaussianNoise
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, Deconv2D, UpSampling2D
from keras.layers.recurrent import LSTM
from keras.regularizers import *
from keras.layers.normalization import *
from keras.optimizers import *
from keras.datasets import mnist
import matplotlib.pyplot as plt
import seaborn as sns
import cPickle, random, sys, keras
from keras.models import Model
from IPython import display

from keras.utils import np_utils
from tqdm import tqdm

# 简单的来说，就是给定一个噪声z的输入，通过生成器的变换把噪声的概率分布空间尽可能的去拟合真实数据的分布空间。

# 在这里，我们把生成器看的目标看成是要“以假乱真”，判别器的目标是要“明辩真假”。

class Gan():
    def build_generator(self):
        # 生成器； 生成器的输入是一个100维服从高斯分布的向量，输出是一张28*28*1的图片。
        model = Sequential()

        model.add(Dense(256, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(Reshape(self.img_shape))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):
        # 判别器； 判别器的输入是一张28*28*1的图片和一个一维的真假标签,1代表是真实世界图片,0代表的的生成模型生成的图片。
        model = Sequential()

        model.add(Flatten(input_shape=self.img_shape))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    # 判别器discriminator只训练判别器的参数;生成器的训练是把生成器和判别器两个网络连在一起,但是冻结判别器的学习率,一起组成combined.
    # 用的都是binary_crossentropy二分类的交叉熵作为损失函数.

    def __init__(self):
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.summary()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()
        self.generator.summary ()
        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        validity = self.discriminator(img)
        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, validity)
        self.combined.summary()
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    # 先加载数据集,然后每一次训练从数据集里面随机选取一张图片进行训练,训练的时候,真实图片对应的标签是valid=1,生成器生成的图片对应的标签是fake=0；
    # 训练的时候,先训练dloss,dloss由真实世界图片和生成图片以及其标签进行训练.在训练判别器的时候,真实世界图片对应真实的标签valid,
    # 生成的图片对应fake标签,也就是让判别器"明辨真假"的过程.在训练生成器的时候,我们输入高斯噪声和噪声标签,等于是告诉生成对抗网络,
    # 我给你一个"假的"图片,但是是"真的"标签,也就是我们让生成器以假乱真的过程.不断的在"明辨真假"和"以假乱真"的两个过程不断迭代训练,
    # 最终,生成器可以很好的"以假乱真",判别器可以很好的"明辨真假".当我们把生成器的图片给"人"看的时候,人就会被"以假乱真"了。
    def train(self, epochs, batch_size=128, sample_interval=50):

        # Load the dataset
        (X_train, _), (_, _) = mnist.load_data()

        # Rescale -1 to 1
        X_train = X_train / 127.5 - 1.
        X_train = np.expand_dims(X_train, axis=3)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Generate a batch of new images
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Train the generator (to have the discriminator label samples as valid)
            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch)

def main():
    pass


if __name__ == '__main__':
    main()

# https://github.com/osh/KerasGAN/blob/master/MNIST_CNN_GAN_v2.ipynb
# https://mp.weixin.qq.com/s/RypciNKUKW0aoYRjU0w5TQ