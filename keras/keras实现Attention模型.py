#!/usr/bin/env python
# coding: utf-8

# # Attention 模型
# 
# 将自然语言时间，如 'quarter after 3 pm'转换为格式化时间'15:15'

# ## 导入依赖包

# In[1]:


# Imports
from keras.layers import Bidirectional, Concatenate, Permute, Dot, Input, LSTM, Multiply, Reshape
from keras.layers import RepeatVector, Dense, Activation, Lambda
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.models import load_model, Model
from keras.callbacks import LearningRateScheduler
import keras.backend as K

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np

import random
import math
import json


# ## 数据集
# 
# 数据集是使用一些简单规则创建的。 它并不详尽，但提出了一些非常好的挑战。
# 
# 该数据集包含在[Github存储库](https://github.com/Choco31415/Attention_Network_With_Keras/blob/master/data/Time%20Dataset.json)中。
# 
# 下面列出了一些示例数据对：
# 
# ['48 min before 10 a.m', '09:12']  
# ['t11:36', '11:36']  
# ["nine o'clock forty six p.m", '21:46']  
# ['2:59p.m.', '14:59']  
# ['23 min after 20 p.m.', '20:23']  
# ['46 min after seven p.m.', '19:46']  
# ['10 before nine pm', '20:50']  
# ['3.20', '03:20']  
# ['7.57', '07:57']  
# ['six hours and fifty five am', '06:55']  

# In[2]:


with open('data/Time Dataset.json','r') as f:
    dataset = json.loads(f.read())
with open('data/Time Vocabs.json','r') as f:
    human_vocab, machine_vocab = json.loads(f.read())
    
human_vocab_size = len(human_vocab)
machine_vocab_size = len(machine_vocab)

# 训练数据量
m = len(dataset)


# 接下来，让我们定义一些通用的辅助方法。 它们用于帮助标记数据。

# In[3]:


def preprocess_data(dataset, human_vocab, machine_vocab, Tx, Ty):
    """
    一种标记数据的方法。
    
     输入：
     数据集-句子数据对的列表。
    human_vocab - A dictionary of tokens (char) to id's.
    machine_vocab - A dictionary of tokens (char) to id's.
    Tx - X data size
    Ty - Y data size
    
    Outputs:
    X - Sparse tokens for X data
    Y - Sparse tokens for Y data
    Xoh - One hot tokens for X data
    Yoh - One hot tokens for Y data
    """
    
    # 源数据
    m = len(dataset)
    
    # 初始化
    X = np.zeros([m, Tx], dtype='int32')
    Y = np.zeros([m, Ty], dtype='int32')
    
    # 预处理
    for i in range(m):
        data = dataset[i]
        X[i] = np.array(tokenize(data[0], human_vocab, Tx))
        Y[i] = np.array(tokenize(data[1], machine_vocab, Ty))
    
    # Expand one hots
    Xoh = oh_2d(X, len(human_vocab))
    Yoh = oh_2d(Y, len(machine_vocab))
    
    return (X, Y, Xoh, Yoh)
    
def tokenize(sentence, vocab, length):
    """
    输出输入的一系列token对应的id列表
    
    支持 <pad> 和 <unk>.
    
    Inputs:
    sentence - 一系列tokens
    vocab - token到id的映射字典
    length - 允许的最大tokens数
    
    Outputs:
    tokens - 
    """
    tokens = [0]*length
    for i in range(length):
        char = sentence[i] if i < len(sentence) else "<pad>"
        char = char if (char in vocab) else "<unk>"
        tokens[i] = vocab[char]
        
    return tokens

def ids_to_keys(sentence, vocab):
    """
    将一系列ID转换为字典的键。
    """
    reverse_vocab = {v: k for k, v in vocab.items()}
    
    return [reverse_vocab[id] for id in sentence]

def oh_2d(dense, max_value):
    """
    为二维输入密集数组（dense array）创建一个热数组（one hot array）。
    """
    # 初始化
    oh = np.zeros(np.append(dense.shape, [max_value]))
    
    # 设置正确的索引
    ids1, ids2 = np.meshgrid(np.arange(dense.shape[0]), np.arange(dense.shape[1]))
    
    oh[ids1.flatten(), ids2.flatten(), dense.flatten('F').astype(int)] = 1
    
    return oh


# 我们的下一个目标是使用词汇表对数据进行标记。

# In[4]:


Tx = 41 # x对应的最大长度
Ty = 5 # y对应的最大长度
X, Y, Xoh, Yoh = preprocess_data(dataset, human_vocab, machine_vocab, Tx, Ty)

# 在训练和测试之间分配数据80:20
train_size = int(0.8*m)
Xoh_train = Xoh[:train_size]
Yoh_train = Yoh[:train_size]
Xoh_test = Xoh[train_size:]
Yoh_test = Yoh[train_size:]


# 为谨慎起见，让我们检查一下代码是否有效：

# In[5]:


i = 4
print("Input data point " + str(i) + ".")
print("")
print("The data input is: " + str(dataset[i][0]))
print("The data output is: " + str(dataset[i][1]))
print("")
print("The tokenized input is:" + str(X[i]))
print("The tokenized output is: " + str(Y[i]))
print("")
print("The one-hot input is:", Xoh[i])
print("The one-hot output is:", Yoh[i])


# ## Model
# 
# 我们的下一个目标是定义我们的模型。 重要的部分将是定义注意机制，然后确保正确应用它。
# 
# 定义一些模型元数据：

# In[6]:


layer1_size = 32
layer2_size = 64 # Attention 层


# 接下来的两个代码段定义了注意机制。 这分为两个部分：
# 
# * 计算上下文
# * 创建关注层
# 
# 注意力网络会在每个输出时间步长上注意输入的某些部分。 注意表示哪些输入与当前输出步骤最相关。 输入步骤的注意权重（如果相关）将为〜1，否则为〜0。 上下文是“输入摘要”。
# 
# 因此，要求是。 注意矩阵应具有形状 $(T_x)$ 且总和为1。此外，对于每个时间步长，上下文的计算方式都应相同。 除此之外，还有一些灵活性。 该笔记本通过以下方式进行计算：
# 
# $$
# attention = Softmax(Dense(Dense(x, y_{t-1})))
# $$
# <br/>
# $$
# context = \sum_{i=1}^{m} ( attention_i * x_i )
# $$
# 
# $y_0$ 默认是 $\vec{0}$.
# 
# 

# In[7]:


# 全局定义一部分关注层，以便为每个关注步骤共享相同的层。
def softmax(x):
    return K.softmax(x, axis=1)

at_repeat = RepeatVector(Tx)
at_concatenate = Concatenate(axis=-1)
at_dense1 = Dense(8, activation="tanh")
at_dense2 = Dense(1, activation="relu")
at_softmax = Activation(softmax, name='attention_weights')
at_dot = Dot(axes=1)

def one_step_of_attention(h_prev, a):
    """
    获取上下文
    
    Input:
    h_prev - RNN层的先前隐藏状态 (m, n_h)
    a - Input data, possibly processed (m, Tx, n_a)
    
    Output:
    context - Current context (m, Tx, n_a)
    """
    # 重复向量以匹配a的尺寸
    h_repeat = at_repeat(h_prev)
    # 计算 attention weights
    i = at_concatenate([a, h_repeat])
    i = at_dense1(i)
    i = at_dense2(i)
    attention = at_softmax(i)
    # 计算 the context
    context = at_dot([attention, a])
    
    return context


# In[8]:


def attention_layer(X, n_h, Ty):
    """
    创建一个 attention 层
    
    Input:
    X - Layer input (m, Tx, x_vocab_size)
    n_h - Size of LSTM 隐藏层
    Ty - 输出顺序中的时间步
    
    Output:
    output - The output of the attention layer (m, Tx, n_h)
    """    
    # 定义LSTM层的默认状态
    h = Lambda(lambda X: K.zeros(shape=(K.shape(X)[0], n_h)))(X)
    c = Lambda(lambda X: K.zeros(shape=(K.shape(X)[0], n_h)))(X)
    # Messy, but the alternative is using more Input()
    
    at_LSTM = LSTM(n_h, return_state=True)
    
    output = []
              
    # 对每个输出时间步长运行注意力步长和RNN
    for _ in range(Ty):
        context = one_step_of_attention(h, X)
        
        h, _, c = at_LSTM(context, initial_state=[h, c])
        
        output.append(h)
        
    return output


# 示例模型的组织如下：:
# 
# 1. BiLSTM
# 2. Attention Layer
#     * 输出激活的 Ty lists.
# 3. Dense
#     * 必须转换 attention layer's output 到正确尺寸

# In[9]:


layer3 = Dense(machine_vocab_size, activation=softmax)

def get_model(Tx, Ty, layer1_size, layer2_size, x_vocab_size, y_vocab_size):
    """
    创建 model.
    
    input:
    Tx - Number of x timesteps
    Ty - Number of y timesteps
    size_layer1 - BiLSTM 神经元数
    size_layer2 - attention LSTM 隐藏层神经元数
    x_vocab_size - Number of possible token types for x
    y_vocab_size - Number of possible token types for y
    
    Output:
    model - A Keras Model.
    """
    
    # Create layers one by one
    X = Input(shape=(Tx, x_vocab_size))
    
    a1 = Bidirectional(LSTM(layer1_size, return_sequences=True), merge_mode='concat')(X)

    a2 = attention_layer(a1, layer2_size, Ty)
    
    a3 = [layer3(timestep) for timestep in a2]
        
    # Create Keras model
    model = Model(inputs=[X], outputs=a3)
    
    return model


# 从这里开始的步骤是创建模型并对其进行训练。

# In[10]:


# 获取模型实例
model = get_model(Tx, Ty, layer1_size, layer2_size, human_vocab_size, machine_vocab_size)


# In[11]:


# Create optimizer
opt = Adam(lr=0.05, decay=0.04, clipnorm=1.0)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])


# In[12]:


# 按时间步对输出进行分组，而不是示例
outputs_train = list(Yoh_train.swapaxes(0,1))


# In[13]:


# 训练时间
# 四核CPU上需要几分钟
model.fit([Xoh_train], outputs_train, epochs=30, batch_size=100)


# ## 评估Evaluation
# 
# 最终的训练损失应在0.02到0.5的范围内
# 
# 测试损失应处于相似的水平。

# In[14]:


# 评估测试性能
outputs_test = list(Yoh_test.swapaxes(0,1))
score = model.evaluate(Xoh_test, outputs_test) 
print('Test loss: ', score[0])


# 现在，我们已经创建了这个模型，让我们看看它是如何工作的。
# 
# 以下代码找到一个随机示例，并通过我们的模型运行它。

# In[15]:


# 让我们直观地检查模型输出。
import random as random

i = random.randint(0, m)

def get_prediction(model, x):
    prediction = model.predict(x)
    max_prediction = [y.argmax() for y in prediction]
    str_prediction = "".join(ids_to_keys(max_prediction, machine_vocab))
    return (max_prediction, str_prediction)

max_prediction, str_prediction = get_prediction(model, Xoh[i:i+1])

print("Input: " + str(dataset[i][0]))
print("Tokenized: " + str(X[i]))
print("Prediction: " + str(max_prediction))
print("Prediction text: " + str(str_prediction))


# 最后但并非最不重要的是，注意力网络的所有介绍都需要进行一些浏览。
# 
# 下图显示了在写每个字母时模型关注的输入内容。

# In[16]:


i = random.randint(0, m)

def plot_attention_graph(model, x, Tx, Ty, human_vocab, layer=7):
    # Process input
    tokens = np.array([tokenize(x, human_vocab, Tx)])
    tokens_oh = oh_2d(tokens, len(human_vocab))
    
    # Monitor model layer
    layer = model.layers[layer]
    
    layer_over_time = K.function(model.inputs, [layer.get_output_at(t) for t in range(Ty)])
    layer_output = layer_over_time([tokens_oh])
    layer_output = [row.flatten().tolist() for row in layer_output]
    
    # Get model output
    prediction = get_prediction(model, tokens_oh)[1]
    
    # Graph the data
    fig = plt.figure()
    fig.set_figwidth(20)
    fig.set_figheight(1.8)
    ax = fig.add_subplot(111)
    
    plt.title("Attention Values per Timestep")
    
    plt.rc('figure')
    cax = plt.imshow(layer_output, vmin=0, vmax=1)
    fig.colorbar(cax)
    
    plt.xlabel("Input")
    ax.set_xticks(range(Tx))
    ax.set_xticklabels(x)
    
    plt.ylabel("Output")
    ax.set_yticks(range(Ty))
    ax.set_yticklabels(prediction)
    
    plt.show()
    
plot_attention_graph(model, dataset[i][0], Tx, Ty, human_vocab)


# In[17]:


model.save('model.h5')
model.summary()


# In[ ]:




