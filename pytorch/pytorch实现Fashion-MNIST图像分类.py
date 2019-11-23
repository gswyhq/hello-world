#!/usr/bin/env python
# coding: utf-8

# 代码来源：https://github.com/zergtant/pytorch-handbook/blob/master/chapter5/5.3-Fashion-MNIST.ipynb


import torch,math
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  
import torchvision.datasets as dsets
from torch.utils.data import Dataset, DataLoader

import torch.nn.functional as F
import torch.nn as NN
torch.__version__


# # Fashion MNIST进行分类 
# ## Fashion MNIST 介绍
# Fashion MNIST数据集 是kaggle上提供的一个图像分类入门级的数据集，其中包含10个类别的70000个灰度图像。如图所示，这些图片显示的是每件衣服的低分辨率(28×28像素)
# 
# 数据集的下载和介绍：[地址](https://www.kaggle.com/zalando-research/fashionmnist/)
# Fashion MNIST数据集 是kaggle上提供的一个图像分类入门级的数据集，其中包含10个类别的70000个灰度图像
# wget -c -t 0 https://gitlab.aiacademy.tw/aai/examples/raw/c13d5198f80d4763216cafa63982ec58d9116365/Fashion-Mnist/fashion-mnist_train.csv
# wget -c -t 0 https://gitlab.aiacademy.tw/aai/examples/raw/c13d5198f80d4763216cafa63982ec58d9116365/Fashion-Mnist/fashion-mnist_test.csv
# http://yann.lecun.com/exdb/mnist/
# wget -c -t 0 http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
# wget -c -t 0 http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz

# 
# Fashion MNIST的目标是作为经典MNIST数据的替换——通常被用作计算机视觉机器学习程序的“Hello, World”。
# 
# MNIST数据集包含手写数字(0-9等)的图像，格式与我们将在这里使用的衣服相同，MNIST只有手写的0-1数据的复杂度不高，所以他只能用来做“Hello, World”
# 
# 而Fashion MNIST 的由于使用的是衣服的数据，比数字要复杂的多，并且图片的内容也会更加多样性，所以它是一个比常规MNIST稍微更具挑战性的问题。
# 
# Fashion MNIST这个数据集相对较小，用于验证算法是否按预期工作。它们是测试和调试代码的好起点。
# 
# ## 数据集介绍
# 
# ### 分类
# ```
# 0 T-shirt/top
# 1 Trouser
# 2 Pullover
# 3 Dress
# 4 Coat
# 5 Sandal
# 6 Shirt
# 7 Sneaker
# 8 Bag
# 9 Ankle boot 
# ```
# ### 格式
# 
# fashion-mnist_test.csv
# 
# fashion-mnist_train.csv
# 
# 存储的训练的数据和测试的数据，格式如下：
# 
# label是分类的标签
# pixel1-pixel784是每一个像素代表的值 因为是灰度图像，所以是一个0-255之间的数值。
# 
# 为什么是784个像素？ 28 * 28 = 784
# 
# ### 数据提交
# 
# Fashion MNIST不需要我们进行数据的提交，数据集中已经帮助我们将 训练集和测试集分好了，我们只需要载入、训练、查看即可，所以Fashion MNIST 是一个非常好的入门级别的数据集
# 

# In[2]:


#指定数据目录
DATA_PATH=Path('/home/gswyhq/data/Fashion-Mnist')


# In[3]:


train = pd.read_csv(DATA_PATH / "fashion-mnist_train.csv");
train.head(10)


# In[4]:


test = pd.read_csv(DATA_PATH / "fashion-mnist_test.csv");
test.head(10)


# In[5]:


train.max()


# ubyte文件标识了数据的格式
# 
# 其中idx3的数字表示数据维度。也就是图像为3维，
# idx1 标签维1维。
# 
# 具体格式详解：http://yann.lecun.com/exdb/mnist/

# In[12]:


import struct
from PIL import Image 

with open(str(DATA_PATH / "train-images-idx3-ubyte"), 'rb') as file_object:  
    header_data=struct.unpack(">4I",file_object.read(16))
    print(header_data)


# In[10]:


with open(str(DATA_PATH / "train-labels-idx1-ubyte"), 'rb') as file_object:
    header_data=struct.unpack(">2I",file_object.read(8))
    print(header_data)


# In[13]:


with open(str(DATA_PATH / "train-images-idx3-ubyte"), 'rb') as file_object:
    raw_img = file_object.read(28*28)
    img = struct.unpack(">784B",raw_img)
    image = np.asarray(img)
    image = image.reshape((28,28))
    print(image.shape)
    plt.imshow(image,cmap = plt.cm.gray)
    plt.show()


# In[14]:


with open(str(DATA_PATH / "train-labels-idx1-ubyte"), 'rb') as file_object:
    raw_img = file_object.read(1)
    label = struct.unpack(">B",raw_img)
    print(label)


# 这里好像有点错误，显示的错位了，但是我的确是按照格式进行处理的。这种格式处理起来比较复杂，并且数据集中的csv直接给出了每个像素的值，所以这里我们可以直接使用csv格式的数据。

# ## 数据加载
# 
# 为了使用pytorch的dataloader进行数据的加载，需要先创建一个自定义的dataset

# In[15]:


class FashionMNISTDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        data = pd.read_csv(csv_file)
        self.X = np.array(data.iloc[:, 1:]).reshape(-1, 1, 28, 28).astype(float)
        self.Y = np.array(data.iloc[:, 0]);
        del data;  #结束data对数据的引用,节省空间
        self.len=len(self.X)

    def __len__(self):
        #return len(self.X)
        return self.len
        
    
    def __getitem__(self, idx):
        item = self.X[idx]
        label = self.Y[idx]
        return (item, label)


# 对于自定义的数据集，只需要实现三个函数：
# 
# `__init__`： 初始化函数主要用于数据的加载，这里直接使用pandas将数据读取为dataframe，然后将其转成numpy数组来进行索引
# 
# `__len__`： 返回数据集的总数，pytorch里面的datalorder需要知道数据集的总数的
# 
# `__getitem__`：会返回单张图片，它包含一个index，返回值为样本及其标签。
# 
# 创建训练和测试集

# In[16]:


train_dataset = FashionMNISTDataset(csv_file=str(DATA_PATH / "fashion-mnist_train.csv"))
test_dataset = FashionMNISTDataset(csv_file=str(DATA_PATH / "fashion-mnist_test.csv"))


# 在使用Pytorch的DataLoader读取数据之前，需要指定一个batch size 这也是一个超参数，涉及到内存的使用量，如果出现OOM的错误则要减小这个数值，一般这个数值都为2的幂或者2的倍数。

# In[17]:


#因为是常量，所以大写，需要说明的是，这些常量建议都使用完整的英文单词，减少歧义
BATCH_SIZE=16 # 这个batch 可以在M250的笔记本显卡中进行训练，不会oom，个人笔记本10G内存，就改成了16


# 我们接着使用dataloader模块来使用这些数据

# In[18]:


train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=BATCH_SIZE,
                                           shuffle=True) # shuffle 标识要打乱顺序


# In[19]:


test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                           batch_size=BATCH_SIZE,
                                           shuffle=False) # shuffle 标识要打乱顺序，测试集不需要打乱


# 查看一下数据

# In[20]:


a=iter(train_loader)
data=next(a)
img=data[0][0].reshape(28,28)
data[0][0].shape,img.shape


# In[21]:


plt.imshow(img,cmap = plt.cm.gray)
plt.show()


# 这回看着就没问题了，是一个完整的图了，所以我们还是用csv吧
# 
# ## 创建网络
# 
# 三层的简单的CNN网络

# In[22]:


class CNN(NN.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = NN.Sequential(   
            NN.Conv2d(1, 16, kernel_size=5, padding=2),
            NN.BatchNorm2d(16), 
            NN.ReLU()) #16, 28, 28
        self.pool1=NN.MaxPool2d(2) #16, 14, 14
        self.layer2 = NN.Sequential(
            NN.Conv2d(16, 32, kernel_size=3),
            NN.BatchNorm2d(32),
            NN.ReLU())#32, 12, 12
        self.layer3 = NN.Sequential(
            NN.Conv2d(32, 64, kernel_size=3),
            NN.BatchNorm2d(64),
            NN.ReLU()) #64, 10, 10
        self.pool2=NN.MaxPool2d(2)  #64, 5, 5
        self.fc = NN.Linear(5*5*64, 10)
    def forward(self, x):
        out = self.layer1(x)
        #print(out.shape)
        out=self.pool1(out)
        #print(out.shape)
        out = self.layer2(out)
        #print(out.shape)
        out=self.layer3(out)
        #print(out.shape)
        out=self.pool2(out)
        #print(out.shape)
        out = out.view(out.size(0), -1)
        #print(out.shape)
        out = self.fc(out)
        return out


# 以上代码看起来很简单。这里面都是包含的数学的含义。我们只讲pytorch相关的：在函数里使用torch.nn提供的模块来定义各个层，在每个卷积层后使用了批次的归一化和RELU激活并且在每一个操作分组后面进行了pooling的操作（减少信息量，避免过拟合），后我们使用了全连接层来输出10个类别。
# 
# view函数用来改变输出值矩阵的形状来匹配最后一层的维度。

# In[23]:


cnn = CNN();
#可以通过以下方式验证，没报错说明没问题，
cnn(torch.rand(1,1,28,28))


# In[24]:


#打印下网络，做最后的确认
print(cnn)


# 从定义模型开始就要指定模型计算的位置，CPU还是GPU，所以需要加另外一个参数

# In[25]:


DEVICE=torch.device("cpu")
if torch.cuda.is_available():
        DEVICE=torch.device("cuda")
print(DEVICE)


# In[26]:


#先把网络放到gpu上
cnn=cnn.to(DEVICE)


# ## 损失函数
# 多分类因为使用Softmax回归将神经网络前向传播得到的结果变成概率分布 所以使用交叉熵损失。
# 在pytorch中 
# NN.CrossEntropyLoss 是将 `nn.LogSoftmax()` 和 `nn.NLLLoss()`进行了整合，[CrossEntropyLoss](https://pytorch.org/docs/stable/nn.html#crossentropyloss) ,我们也可以分开来写使用两步计算，这里为了方便直接一步到位
# 

# In[27]:


#损失函数也需要放到GPU中
criterion = NN.CrossEntropyLoss().to(DEVICE)


# ## 优化器
# Adam 优化器：简单，暴力，最主要还是懒

# In[28]:


#另外一个超参数，学习率
LEARNING_RATE=0.01


# In[29]:


#优化器不需要放GPU
optimizer = torch.optim.Adam(cnn.parameters(), lr=LEARNING_RATE)


# ## 开始训练

# In[30]:


#另外一个超参数，指定训练批次
TOTAL_EPOCHS=50


# In[ ]:

#记录损失函数
losses = [];
for epoch in range(TOTAL_EPOCHS):
    for i, (images, labels) in enumerate(train_loader):
        images = images.float().to(DEVICE)
        labels = labels.to(DEVICE)
        #清零
        optimizer.zero_grad()
        outputs = cnn(images)
        #计算损失函数
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        losses.append(loss.cpu().data.item());
        if (i+1) % 100 == 0:
            print ('Epoch : %d/%d, Iter : %d/%d,  Loss: %.4f'%(epoch+1, TOTAL_EPOCHS, i+1, len(train_dataset)//BATCH_SIZE, loss.data.item()))
            

# ## 训练后操作
# ### 可视化损失函数

# In[28]:


plt.xkcd();
plt.xlabel('Epoch #');
plt.ylabel('Loss');
plt.plot(losses);
plt.show();


# ### 保存模型 

# In[29]:


torch.save(cnn.state_dict(), "fm-cnn3.pth")
# 加载用这个
#cnn.load_state_dict(torch.load("fm-cnn3.pth"))


# ## 模型评估
# 
# 模型评估就是使用测试集对模型进行的评估，应该是添加到训练中进行了，这里为了方便说明直接在训练完成后评估了

# In[30]:


cnn.eval()
correct = 0
total = 0
for images, labels in test_loader:
    images = images.float().to(DEVICE)
    outputs = cnn(images).cpu()
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()
print('准确率: %.4f %%' % (100 * correct / total))


# 模型评估的步骤如下：
# 1. 将网络的模式改为eval。
# 2. 将图片输入到网络中得到输出。
# 3. 通过取出one-hot输出的最大值来得到输出的 标签。
# 4. 统计正确的预测值。

# ## 进一步优化

# In[31]:

#修改学习率和批次
cnn.train()
LEARNING_RATE=LEARNING_RATE / 10
TOTAL_EPOCHS=20
optimizer = torch.optim.Adam(cnn.parameters(), lr=0.001)
losses = [];
for epoch in range(TOTAL_EPOCHS):
    for i, (images, labels) in enumerate(train_loader):
        images = images.float().to(DEVICE)
        labels = labels.to(DEVICE)
        #清零
        optimizer.zero_grad()
        outputs = cnn(images)
        #计算损失函数
        #损失函数直接放到CPU中，因为还有其他的计算
        loss = criterion(outputs, labels).cpu()
        loss.backward()
        optimizer.step()
        losses.append(loss.data.item());
        if (i+1) % 100 == 0:
            print ('Epoch : %d/%d, Iter : %d/%d,  Loss: %.4f'%(epoch+1, TOTAL_EPOCHS, i+1, len(train_dataset)//BATCH_SIZE, loss.data.item()))
            

# 可视化一下损失

# In[32]:


plt.xkcd();
plt.xlabel('Epoch #');
plt.ylabel('Loss');
plt.plot(losses);
plt.show();


# ## 再次进行评估

# In[33]:


cnn.eval()
correct = 0
total = 0
for images, labels in test_loader:
    images = images.float().to(DEVICE)
    outputs = cnn(images).cpu()
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()
print('准确率: %.4f %%' % (100 * correct / total))


# In[34]:


#修改学习率和批次
cnn.train()
LEARNING_RATE=LEARNING_RATE / 10
TOTAL_EPOCHS=10
optimizer = torch.optim.Adam(cnn.parameters(), lr=0.001)
losses = [];
for epoch in range(TOTAL_EPOCHS):
    for i, (images, labels) in enumerate(train_loader):
        images = images.float().to(DEVICE)
        labels = labels.to(DEVICE)
        #清零
        optimizer.zero_grad()
        outputs = cnn(images)
        #计算损失函数
        #损失函数直接放到CPU中，因为还有其他的计算
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        losses.append(loss.cpu().data.item());
        if (i+1) % 100 == 0:
            print ('Epoch : %d/%d, Iter : %d/%d,  Loss: %.4f'%(epoch+1, TOTAL_EPOCHS, i+1, len(train_dataset)//BATCH_SIZE, loss.data.item()))
            


# In[35]:


plt.xkcd();
plt.xlabel('Epoch #');
plt.ylabel('Loss');
plt.plot(losses);
plt.show();


# In[36]:


cnn.eval()
correct = 0
total = 0
for images, labels in test_loader:
    images = images.float().to(DEVICE)
    outputs = cnn(images).cpu()
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()
print('准确率: %.4f %%' % (100 * correct / total))


# 损失小了，但是准确率没有提高，这就说明已经接近模型的瓶颈了，如果再要进行优化，就需要修改模型了。另外还有一个判断模型是否到瓶颈的标准，就是看损失函数，最后一次的训练的损失函数明显的没有下降的趋势，只是在震荡，这说明已经没有什么优化的空间了。
# 
# 通过简单的操作，我们也能够看到Adam优化器的暴力性，我们只要简单的修改学习率就能够达到优化的效果，Adam优化器的使用一般情况下是首先使用0.1进行预热，然后再用0.01进行大批次的训练，最后使用0.001这个学习率进行收尾，再小的学习率一般情况就不需要了。
# 
# ## 总结
# 最后我们再总结一下几个超参数:
# 
# `BATCH_SIZE`: 批次数量，定义每次训练时多少数据作为一批，这个批次需要在dataloader初始化时进行设置，并且需要这对模型和显存进行配置，如果出现OOM有线减小，一般设为2的倍数
# 
# `DEVICE`：进行计算的设备，主要是CPU还是GPU
# 
# `LEARNING_RATE`：学习率，反向传播时使用
# 
# `TOTAL_EPOCHS`：训练的批次，一般情况下会根据损失和准确率等阈值
# 
# 其实优化器和损失函数也算超参数，这里就不说了

# In[ ]:




