#!/usr/bin/env python
# coding: utf-8

# 来源： https://github.com/fchollet/deep-learning-with-python-notebooks.gi
# In[1]:


import keras
keras.__version__


# 使用词嵌入
#
# 将单词与向量相关联还有另一种常用的强大方法,就是使用密集的词向量(word vector),也叫词嵌入(word embedding)。
# one-hot 编码得到的向量是二进制的、稀疏的(绝大部分元素都是 0)、维度很高的(维度大小等于词表中的单词个数),
# 而词嵌入是低维的浮点数向量(即密集向量,与稀疏向量相对),参见图 6-2。与 one-hot 编码得到的词向量不同,词嵌入是从数据中学习得到的。
#
# 常见的词向量维度是 256、512 或 1024(处理非常大的词表时)。
# 与此相对,one-hot 编码的词向量维度通常为 20 000 或更高(对应包含 20 000 个标记的词表)。
# 因此,词向量可以将更多的信息塞入更低的维度中。

# ![word embeddings vs. one hot encoding](https://s3.amazonaws.com/book.keras.io/img/ch6/word_embeddings.png)

# 获取词嵌入有两种方法。
#  在完成主任务(比如文档分类或情感预测)的同时学习词嵌入。在这种情况下,一开始是随机的词向量,然后对这些词向量进行学习,其学习方式与学习神经网络的权重相同。
# 在不同于待解决问题的机器学习任务上预计算好词嵌入,然后将其加载到模型中。这些词嵌入叫作预训练词嵌入(pretrained word embedding)。


# 利用 Embedding 层学习词嵌入

# 要将一个词与一个密集向量相关联,最简单的方法就是随机选择向量。
# 这种方法的问题在于,得到的嵌入空间没有任何结构。例如,accurate 和 exact 两个词的嵌入可能完全不同,
# 尽管它们在大多数句子里都是可以互换的 a 。深度神经网络很难对这种杂乱的、非结构化的嵌入空间进行学习。

# 说得更抽象一点,词向量之间的几何关系应该表示这些词之间的语义关系。词嵌入的作用应该是将人类的语言映射到几何空间中。
# 例如,在一个合理的嵌入空间中,同义词应该被嵌入到相似的词向量中,
# 一般来说,任意两个词向量之间的几何距离(比如 L2 距离)应该和这两个词的语义距离有关(表示不同事物的词被嵌入到相隔很远的点,
# 而相关的词则更加靠近)。除了距离,你可能还希望嵌入空间中的特定方向也是有意义的。为了更清楚地说明这一点,我们来看一个具体示例。


# 在真实的词嵌入空间中,常见的有意义的几何变换的例子包括“性别”向量和“复数”向量。
# 例如,将 king (国王)向量加上 female (女性)向量,得到的是 queen (女王)向量。将 king (国王)
# 向量加上 plural(复数)向量,得到的是 kings 向量。词嵌入空间通常具有几千个这种可解释的、
# 并且可能很有用的向量。
# 有没有一个理想的词嵌入空间,可以完美地映射人类语言,并可用于所有自然语言处理任
# 务?可能有,但我们尚未发现。此外,也不存在人类语言(human language)这种东西。
# 世界上有许多种不同的语言,而且它们不是同构的,因为语言是特定文化和特定环境的反射。但从更
# 实际的角度来说,一个好的词嵌入空间在很大程度上取决于你的任务。英语电影评论情感分析
# 模型的完美词嵌入空间,可能不同于英语法律文档分类模型的完美词嵌入空间,因为某些语义
# 关系的重要性因任务而异。
# 因此,合理的做法是对每个新任务都学习一个新的嵌入空间。幸运的是,反向传播让这种
# 学习变得很简单,而 Keras 使其变得更简单。我们要做的就是学习一个层的权重,这个层就是
# Embedding 层。

# In[3]:


from keras.layers import Embedding

# Embedding 层至少需要两个参数:标记的个数(这里是 1000,即最大单词索引 +1)和嵌入的维度(这里是 64)
embedding_layer = Embedding(1000, 64)


# 最好将 Embedding 层理解为一个字典,将整数索引(表示特定单词)映射为密集向量。
# 它接收整数作为输入,并在内部字典中查找这些整数,然后返回相关联的向量。 Embedding 层实际上是一种字典查找
# 单词索引 -> Embedding层 -> 对应的词向量

# Embedding 层的输入是一个二维整数张量,其形状为 (samples, sequence_length) ,
# 每个元素是一个整数序列。它能够嵌入长度可变的序列,例如,对于前一个例子中的Embedding 层,
# 你可以输入形状为 (32, 10) (32 个长度为 10 的序列组成的批量)或 (64,15) (64 个长度为 15 的序列组成的批量)的批量。
# 不过一批数据中的所有序列必须具有相同的长度(因为需要将它们打包成一个张量),所以较短的序列应该用 0 填充,较长的序列应该被截断。
# 这 个 Embedding 层 返 回 一 个 形 状 为 (samples, sequence_length, embedding_dimensionality) 的三维浮点数张量。
# 然后可以用 RNN 层或一维卷积层来处理这个三维张量
# 
# 将一个 Embedding 层实例化时,它的权重(即标记向量的内部字典)最开始是随机的,与
# 其他层一样。在训练过程中,利用反向传播来逐渐调节这些词向量,改变空间结构以便下游模
# 型可以利用。一旦训练完成,嵌入空间将会展示大量结构,这种结构专门针对训练模型所要解
# 决的问题。
# 我们将这个想法应用于你熟悉的 IMDB 电影评论情感预测任务。首先,我们需要快速准备
# 数据。将电影评论限制为前 10 000 个最常见的单词(第一次处理这个数据集时就是这么做的),
# 然后将评论长度限制为只有 20 个单词。对于这 10 000 个单词,网络将对每个词都学习一个 8
# 维嵌入,将输入的整数序列(二维整数张量)转换为嵌入序列(三维浮点数张量),然后将这个
# 张量展平为二维,最后在上面训练一个 Dense 层用于分类。

# 加载 IMDB 数据,准备用于 Embedding 层
from keras.datasets import imdb
from keras import preprocessing

# 作为特征的单词个数
max_features = 10000
# 在这么多单词后截断文本(这些单词都属于前 max_features 个最常见的单词)
maxlen = 20

# 将数据加载为整数列表
# 将从https://s3.amazonaws.com/text-datasets/imdb.npz 下载数据到：.keras/datasets/imdb.npz
# 链接：https://pan.baidu.com/s/1L7rNOHsFsAJSNirWM4ykMw 密码：kjpa

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

# 将整数列表转换成形状为 (samples,maxlen) 的二维整数张量
x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)


# In[5]:


from keras.models import Sequential
from keras.layers import Flatten, Dense

model = Sequential()
# 指定 Embedding 层的最大输入长度,以便后面将嵌入输入展平。 Embedding 层激活的形状为 (samples, maxlen, 8)
model.add(Embedding(10000, 8, input_length=maxlen))

# 将三维的嵌入张量展平成形状为 (samples, maxlen * 8) 的二维张量
model.add(Flatten())

# 在上面添加分类器
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
model.summary()

history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=32,
                    validation_split=0.2)


# 得到的验证精度约为 76%,考虑到仅查看每条评论的前 20 个单词,这个结果还是相当不错
# 的。但请注意,仅仅将嵌入序列展开并在上面训练一个 Dense 层,会导致模型对输入序列中的
# 每个单词单独处理,而没有考虑单词之间的关系和句子结构(举个例子,这个模型可能会将 this
# movie is a bomb 和 this movie is the bomb 两条都归为负面评论 a )。更好的做法是在嵌入序列上添
# 加循环层或一维卷积层,将每个序列作为整体来学习特征。

# 使用预训练的词嵌入
# 有时可用的训练数据很少,以至于只用手头数据无法学习适合特定任务的词嵌入。那么应
# 该怎么办?
# 你可以从预计算的嵌入空间中加载嵌入向量(你知道这个嵌入空间是高度结构化的,并且
# 具有有用的属性,即抓住了语言结构的一般特点),而不是在解决问题的同时学习词嵌入。在自
# 然语言处理中使用预训练的词嵌入,其背后的原理与在图像分类中使用预训练的卷积神经网络
# 是一样的:没有足够的数据来自己学习真正强大的特征,但你需要的特征应该是非常通用的,
# 比如常见的视觉特征或语义特征。在这种情况下,重复使用在其他问题上学到的特征,这种做
# 法是有道理的。
# 这种词嵌入通常是利用词频统计计算得出的(观察哪些词共同出现在句子或文档中),用到
# 的技术很多,有些涉及神经网络,有些则不涉及。Bengio 等人在 21 世纪初首先研究了一种思路,
# 就是用无监督的方法计算一个密集的低维词嵌入空间 a ,但直到最有名且最成功的词嵌入方案之
# 一 word2vec 算法发布之后,这一思路才开始在研究领域和工业应用中取得成功。word2vec 算法
# 由 Google 的 Tomas Mikolov 于 2013 年开发,其维度抓住了特定的语义属性,比如性别。
# 有 许 多 预 计 算 的 词 嵌 入 数 据 库, 你 都 可 以 下 载 并 在 Keras 的 Embedding 层 中 使 用。
# word2vec 就是其中之一。另一个常用的是 GloVe(global vectors for word representation,词表示
# 全局向量),由斯坦福大学的研究人员于 2014 年开发。这种嵌入方法基于对词共现统计矩阵进
# 行因式分解。其开发者已经公开了数百万个英文标记的预计算嵌入,它们都是从维基百科数据
# 和 Common Crawl 数据得到的。
# 我们来看一下如何在 Keras 模型中使用 GloVe 嵌入。同样的方法也适用于 word2vec 嵌入或
# 其他词嵌入数据库。这个例子还可以改进前面刚刚介绍过的文本分词技术,即从原始文本开始,
# 一步步进行处理。

# 整合在一起:从原始文本到词嵌入
# 本节的模型与之前刚刚见过的那个类似:将句子嵌入到向量序列中,然后将其展平,最后
# 在上面训练一个 Dense 层。但此处将使用预训练的词嵌入。此外,我们将从头开始,先下载
# IMDB 原始文本数据,而不是使用 Keras 内置的已经预先分词的 IMDB 数据。

# 下载 IMDB 数据的原始文本
# 首先,打开 http://mng.bz/0tIo,下载原始 IMDB 数据集并解压。
# 接下来,我们将训练评论转换成字符串列表,每个字符串对应一条评论。你也可以将评论
# 标签(正面 / 负面)转换成 labels 列表。

# In[6]:


import os

imdb_dir = '/home/gswyhq/data/aclImdb'
train_dir = os.path.join(imdb_dir, 'train')

labels = []
texts = []

for label_type in ['neg', 'pos']:
    dir_name = os.path.join(train_dir, label_type)
    for fname in os.listdir(dir_name):
        if fname[-4:] == '.txt':
            f = open(os.path.join(dir_name, fname))
            texts.append(f.read())
            f.close()
            if label_type == 'neg':
                labels.append(0)
            else:
                labels.append(1)


# 对数据进行分词
# 利用本节前面介绍过的概念,我们对文本进行分词,并将其划分为训练集和验证集。因为
# 预训练的词嵌入对训练数据很少的问题特别有用(否则,针对于具体任务的嵌入可能效果更好),
# 所以我们又添加了以下限制:将训练数据限定为前 200 个样本。因此,你需要在读取 200 个样
# 本之后学习对电影评论进行分类。

# In[7]:


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np

maxlen = 100  # 在 100 个单词后截断评论
training_samples = 200  # 在 200 个样本上训练
validation_samples = 10000  # 在 10 000 个样本上验证
max_words = 10000  # 只考虑数据集中前 10 000 个最常见的单词

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=maxlen)

labels = np.asarray(labels)
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# 将数据划分为训练集和验证集,但首先要打乱数据,因为一开始数据中的样本是排好序的(所有负面评论都在前面,然后是所有正面评论)

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

x_train = data[:training_samples]
y_train = labels[:training_samples]
x_val = data[training_samples: training_samples + validation_samples]
y_val = labels[training_samples: training_samples + validation_samples]


# 下载 GloVe 词嵌入
# 打开 https://nlp.stanford.edu/projects/glove,下载 2014 年英文维基百科的预计算嵌入。这是
# 一个 822 MB 的压缩文件,文件名是 glove.6B.zip,里面包含 400 000 个单词(或非单词的标记)
# 的 100 维嵌入向量。解压文件。 1

# 对嵌入进行预处理
# 我们对解压后的文件(一个 .txt 文件)进行解析,构建一个将单词(字符串)映射为其向
# 量表示(数值向量)的索引。


glove_dir = '/home/ubuntu/data/'

embeddings_index = {}
f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))


# 接下来,需要构建一个可以加载到 Embedding 层中的嵌入矩阵。它必须是一个形状为
# (max_words, embedding_dim) 的矩阵,对于单词索引(在分词时构建)中索引为 i 的单词,

# 这个矩阵的元素 i 就是这个单词对应的 embedding_dim 维向量。注意,索引 0 不应该代表任何
# 单词或标记,它只是一个占位符。

# 准备 GloVe 词嵌入矩阵

embedding_dim = 100

embedding_matrix = np.zeros((max_words, embedding_dim))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if i < max_words:
        if embedding_vector is not None:
            # 嵌入索引( embeddings_index )中找不到的词,其嵌入向量全为 0
            embedding_matrix[i] = embedding_vector


# 定义模型
# 我们将使用与前面相同的模型架构

# In[15]:


from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense

model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()


# 在模型中加载 GloVe 嵌入
# Embedding 层只有一个权重矩阵,是一个二维的浮点数矩阵,其中每个元素 i 是与索引 i
# 相关联的词向量。够简单。将准备好的 GloVe 矩阵加载到 Embedding 层中,即模型的第一层。
# 将预训练的词嵌入加载到 Embedding 层中

model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False


# 此外,需要冻结 Embedding 层(即将其 trainable 属性设为 False )
# ,其原理和预训练的卷
# 积神经网络特征相同,你已经很熟悉了。如果一个模型的一部分是经过预训练的(如 Embedding
# 层),而另一部分是随机初始化的(如分类器),那么在训练期间不应该更新预训练的部分,以
# 避免丢失它们所保存的信息。随机初始化的层会引起较大的梯度更新,会破坏已经学到的特征。


# 训练模型与评估模型
# 编译并训练模型。

# In[17]:


model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])
history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=32,
                    validation_data=(x_val, y_val))
model.save_weights('pre_trained_glove_model.h5')


# 接下来,绘制模型性能随时间的变化

# In[18]:


import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()


# 模型很快就开始过拟合,考虑到训练样本很少,这一点也不奇怪。出于同样的原因,验证
# 精度的波动很大,但似乎达到了接近 60%。
# 
# 注意,你的结果可能会有所不同。训练样本数太少,所以模型性能严重依赖于你选择的
# 200 个样本,而样本是随机选择的。如果你得到的结果很差,可以尝试重新选择 200 个不同的
# 随机样本,你可以将其作为练习(在现实生活中无法选择自己的训练数据)。
# 你也可以在不加载预训练词嵌入、也不冻结嵌入层的情况下训练相同的模型。在这种情况下,
# 你将会学到针对任务的输入标记的嵌入。如果有大量的可用数据,这种方法通常比预训练词嵌
# 入更加强大,但本例只有 200 个训练样本。我们来试一下这种方法
# In[20]:

# 在不使用预训练词嵌入的情况下,训练相同的模型

from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense

model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])
history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=32,
                    validation_data=(x_val, y_val))


# In[22]:


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()


# 验证精度停留在 50% 多一点。因此,在本例中,预训练词嵌入的性能要优于与任务一起学
# 习的嵌入。如果增加样本数量,情况将很快发生变化,你可以把它作为一个练习。
# 最后,我们在测试数据上评估模型。首先,你需要对测试数据进行分词。
# In[24]:


test_dir = os.path.join(imdb_dir, 'test')

labels = []
texts = []

for label_type in ['neg', 'pos']:
    dir_name = os.path.join(test_dir, label_type)
    for fname in sorted(os.listdir(dir_name)):
        if fname[-4:] == '.txt':
            f = open(os.path.join(dir_name, fname))
            texts.append(f.read())
            f.close()
            if label_type == 'neg':
                labels.append(0)
            else:
                labels.append(1)

sequences = tokenizer.texts_to_sequences(texts)
x_test = pad_sequences(sequences, maxlen=maxlen)
y_test = np.asarray(labels)


# 在测试集上评估模型
# In[25]:


model.load_weights('pre_trained_glove_model.h5')
model.evaluate(x_test, y_test)


# 测试精度达到了令人震惊的 56% !只用了很少的训练样本,得到这样的结果很不容易。