#!/usr/bin/env python3
# -*- coding: utf-8 -*-

##############################################################Keras#####################################################################################

# 1.1 数据集加载与预处理
# 首先是导入相关包，然后加载MNIST数据
import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, \
    AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model
from keras.optimizers import Adam

#加载数据
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# #(60000,28,28)
# print('x_shape',x_train.shape)
# #(60000)
# print('y_shape',y_train.shape)

# 然后对数据集进行处理：
# 将数据reshape为(-1,28,28,1)的四维向量，1表示黑白图像(3表示彩色图像)，之后进行归一化，将标签转为one-hot编码。

#数据集处理
x_train=x_train.reshape(-1,28,28,1)/255.0   #reshape为(60000,28,28,1)的四维向量,1表示黑白图像(3表示彩色图像);/255表示归一化
x_test=x_test.reshape(-1,28,28,1)/255
#标签转换为one-hot编码
y_train=np_utils.to_categorical(y_train,num_classes=10)
y_test=np_utils.to_categorical(y_test,num_classes=10)

# 1.2 搭建模型
# 模型结构如下：

model = Sequential()
model.add(Conv2D(6,kernel_size=(5,5),padding='same',strides=(1,1),activation='sigmoid'))
model.add(AveragePooling2D(pool_size=(2,2)))
model.add(Conv2D(16, kernel_size=(5,5), activation='sigmoid'))
model.add(AveragePooling2D(pool_size=(2,2)))
#池化后变成16个4x4的矩阵，然后把矩阵压平变成一维的，一共256个单元
model.add(Flatten())
# 下面就是全连接层了
model.add(Dense(120, activation='sigmoid'))
model.add(Dense(84, activation='sigmoid'))
# softmax激活函数是用于计算该输入图像属于0-9数字的概率
model.add(Dense(10,activation='softmax'))

model.build(x_train.shape)
# 可以使用summary查看模型结构，模型结构如下：
# model.summary()
# Model: "sequential"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  conv2d (Conv2D)             multiple                  156
#
#  average_pooling2d (AverageP  multiple                 0
#  ooling2D)
#
#  conv2d_1 (Conv2D)           multiple                  2416
#
#  average_pooling2d_1 (Averag  multiple                 0
#  ePooling2D)
#
#  flatten (Flatten)           multiple                  0
#
#  dense (Dense)               multiple                  48120
#
#  dense_1 (Dense)             multiple                  10164
#
#  dense_2 (Dense)             multiple                  850
#
# =================================================================
# Total params: 61,706
# Trainable params: 61,706
# Non-trainable params: 0
# _________________________________________________________________

# 1.3 训练模型
# 使用Adam优化器进行加速，以及二元交叉熵损失作为损失函数：

adam=Adam(lr=0.01)
model.compile(loss="categorical_crossentropy", optimizer=adam, metrics=['accuracy'])



# 然后使用fit函数进行训练：

model.fit(x_train,y_train,batch_size=64,epochs=10,validation_split=0.2,shuffle=True)

# 1.4 评估模型
# 可以使用evaluate函数评估模型的准确率和损失：

#评估模型
loss,accuracy=model.evaluate(x_test,y_test)
print('\naccuracy:',np.round_(accuracy*100,5),'%')
print('\ntest loss:',loss)

# 运行的准确率和损失如下：
# accuracy: 98.7 %
# test loss: 0.04736718535423279



####################################################################Pytorch###############################################################################

# 2.1 数据集加载与预处理
# 首先是导入相关包，然后加载MNIST数据集：
from torch.utils import data
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import sys
from torch import nn
import torch.nn.functional as F

#定义加载数据集函数
def load_data_mnist(batch_size):
    '''下载MNIST数据集然后加载到内存中'''
    train_dataset=datasets.MNIST(root='~/.torch',train=True,transform=transforms.ToTensor(),download=True)
    test_dataset=datasets.MNIST(root='~/.torch',train=False,transform=transforms.ToTensor(),download=True)
    return (data.DataLoader(train_dataset,batch_size,shuffle=True),
           data.DataLoader(test_dataset,batch_size,shuffle=False))

#LeNet-5在MNIST数据集上的表现
batch_size=64
train_iter,test_iter=load_data_mnist(batch_size=batch_size)

# 2.2 搭建模型
# 接下来进行搭建模型，模型输入为(-1,1,28,28)【注意这里与keras不同】，然后进行搭建：

class Reshape(torch.nn.Module):
    def forward(self, x):
        return x.view(-1, 1, 28, 28)

#LeNet-5网络结构
net=nn.Sequential(
    Reshape(),nn.Conv2d(1,6,kernel_size=5,padding=2),nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2,stride=2),
    nn.Conv2d(6,16,kernel_size=5),nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2,stride=2),nn.Flatten(),
    nn.Linear(16*5*5,120),nn.Sigmoid(),
    nn.Linear(120,84),nn.Sigmoid(),
    nn.Linear(84,10))

# 搭建完成后对模型检查模型层次：

#检查模型
x=torch.rand(size=(1,1,28,28),dtype=torch.float32)
for layer in net:
    x=layer(x)
    print(layer.__class__.__name__,'output shape:\t',x.shape)

# Reshape output shape:	 torch.Size([1, 1, 28, 28])
# Conv2d output shape:	 torch.Size([1, 6, 28, 28])
# Sigmoid output shape:	 torch.Size([1, 6, 28, 28])
# AvgPool2d output shape:	 torch.Size([1, 6, 14, 14])
# Conv2d output shape:	 torch.Size([1, 16, 10, 10])
# Sigmoid output shape:	 torch.Size([1, 16, 10, 10])
# AvgPool2d output shape:	 torch.Size([1, 16, 5, 5])
# Flatten output shape:	 torch.Size([1, 400])
# Linear output shape:	 torch.Size([1, 120])
# Sigmoid output shape:	 torch.Size([1, 120])
# Linear output shape:	 torch.Size([1, 84])
# Sigmoid output shape:	 torch.Size([1, 84])
# Linear output shape:	 torch.Size([1, 10])

# 2.3 训练模型
# 定义损失函数和优化器，损失函数使用二元交叉熵损失CrossEntropyLoss，优化器采用Adam优化器：

#损失函数
loss_function=nn.CrossEntropyLoss()
#优化器
optimizer=torch.optim.Adam(net.parameters())

# 训练10个批次，训练代码如下：

# 开始训练
num_epochs = 10
train_loss = []
for epoch in range(num_epochs):
    for batch_idx, (x, y) in enumerate(train_iter):
        #         x = x.view(x.size(0), 28 * 28)
        out = net(x)
        y_onehot = F.one_hot(y, num_classes=10).float()  # 转为one-hot编码

        loss = loss_function(out, y_onehot)  # 均方差
        # 清零梯度
        optimizer.zero_grad()
        loss.backward()
        # w' = w -lr * grad
        optimizer.step()
        train_loss.append(loss.item())
        if batch_idx % 10 == 0:
            print(epoch, batch_idx, loss.item())

# 并绘制损失曲线：

#绘制损失曲线
plt.figure(figsize=(8,3))
plt.grid(True,linestyle='--',alpha=0.5)
plt.plot(train_loss,label='loss')
plt.legend(loc="best")
plt.show()

# 2.4 评估模型
# 利用训练好的模型评估测试准确率：

total_correct = 0
for batch_idx, (x, y) in enumerate(test_iter):
    #     x = x.view(x.size(0),28*28)
    out = net(x)
    pred = out.argmax(dim=1)
    correct = pred.eq(y).sum().float().item()
    total_correct += correct

total_num = len(test_iter.dataset)
test_acc = total_correct / total_num
print(total_correct, total_num)
print("test acc:", test_acc)

# 9814.0 10000
# test acc: 0.9814

############################################################区别与联系#######################################################################################
# Keras	PyTorch
# 输入	Keras的输入是(-1,28,28,1)的四维向量，通道放在最后一维上	Pytorch的输入是(-1,1,28,28)，通道在第二个维度上
# 模型搭建	Keras的模型无需表明输入，只需表明输出即可	Pytorch的模型搭建必须标明输入和输出
# 模型训练	Keras利用fit函数进行模型训练，较为简洁	Pytorch利用迭代进行模型训练且梯度清零、误差反馈和梯度更新这三行代码是必不可少的代码
# 评估模型	Keras利用evaluate评估模型	Pytorch利用net网络的输出评估模型

############################################################ pytorch 像 keras 一样 使用： summary #######################################################################################

# 在我们构建一个模型并进行训练的时候，有时候我们希望观察网络的每个层是什么操作、输出维度、模型的总参数量、训练的参数量、网络的占用内存情况。
# 在pytorch下torchsummary包和torchkeras包可以完美又简洁的输出用用pytorch写的网络的相关信息。
# 类似于Keras的model.summary()的功能。

# 方法一：torchsummary
# pip install torch-summary
# 举例
import time
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class FC(nn.Module):
    def __init__(self):
        super().__init__()
        self.liner_1 = nn.Linear(40 * 40, 120)
        self.liner_2 = nn.Linear(120, 84)
        self.liner_3 = nn.Linear(84, 2)

    def forward(self, input):
        x = input.view(-1, 40 * 40)
        x = F.relu(self.liner_1(x))
        x = F.relu(self.liner_2(x))
        x = self.liner_3(x)
        return x


model = FC()
print(model)
summary(model, (3, 40, 40))

# 方法二：torchkeras使用torchkeras打印Pytorch模型结构和基本参数信息
# pip install torchkeras
import torch
from torch import nn
from torchkeras import summary


def create_net():
    net = nn.Sequential()
    net.add_module('linear1', nn.Linear(15, 20))
    net.add_module('relu1', nn.ReLU())
    net.add_module('linear2', nn.Linear(20, 1))
    net.add_module('sigmoid', nn.Sigmoid())
    return net


# 创建模型
net = create_net()
# 使用torchkeras中的summary函数打印模型结构和参数
print(summary(net, input_shape=(15,)))
# 结果如下所示：----------------------------------------------------------------


def main():
    pass


if __name__ == '__main__':
    main()
