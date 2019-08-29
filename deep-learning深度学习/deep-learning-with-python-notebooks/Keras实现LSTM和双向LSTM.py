#!/usr/bin/python3
# coding: utf-8

# https://blog.csdn.net/qq_41853758/article/details/82455837
from keras.layers import Embedding, Flatten, Bidirectional, Dropout, LSTM, Dense
from keras.models import Sequential
from keras.callbacks import EarlyStopping

def lstm_model():
    model = Sequential()
    model.add(Embedding(3800, 32, input_length=380))
    model.add(Dropout(0.2))
    model.add(LSTM(32))
    # 设 LSTM 输入维度为 x_dim， 输出维度为 y_dim，那么参数个数 n 为：
    # n = 4 * ((x_dim + y_dim) * y_dim + y_dim)

    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    print(model.summary())
'''

_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding_1 (Embedding)      (None, 380, 32)           121600    = 3800*32
_________________________________________________________________
dropout_1 (Dropout)          (None, 380, 32)           0         
_________________________________________________________________
lstm_1 (LSTM)                (None, 32)                8320      = 4 * ((32 + 32) * 32 + 32)
_________________________________________________________________
dense_1 (Dense)              (None, 256)               8448      = 32*256+256
_________________________________________________________________
dropout_2 (Dropout)          (None, 256)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 257       = 256+1
=================================================================
Total params: 138,625
Trainable params: 138,625
Non-trainable params: 0
_________________________________________________________________
None

g(t)=ϕ(Wgx*x(t)+Wgh*h(t−1)+bg
i(t)=σ(Wix*x(t)+Wih*h(t−1)+bi
f(t)=σ(Wfx*x(t)+Wfh*h(t−1)+bf
o(t)=σ(Wox*x(t)+Woh*h(t−1)+bo
s(t)=g(t)∗i(t)+s(t−1)∗f(t)
h(t)=s(t)∗o(t)

这里x(t),h(t)分别是我们的输入序列和输出序列。如果我们把x(t)与h(t−1)这两个向量进行合并:
xc(t)=[x(t),h(t−1)]
公式的h(t−1)是上一个状态的隐向量(已设定隐向量长度为10)
x(t)为当前状态的输入(长度为5),那么[x(t),h(t−1)]的长度就是10+5=15了

那么可以上面的方程组可以重写为:

g(t)=ϕ(Wg*xc(t))+bg
i(t)=σ(Wi*xc(t))+bi
f(t)=σ(Wf*xc(t))+bf
o(t)=σ(Wo*xc(t))+bo
s(t)=g(t)∗i(t)+s(t−1)∗f(t)
h(t)=s(t)∗o(t)
其中f(t)被称为忘记门，所表达的含义是决定我们会从以前状态中丢弃什么信息。
i(t),g(t)构成了输入门，决定什么样的新信息被存放在细胞状态中。
o(t)所在位置被称作输出门，决定我们要输出什么值。

设 LSTM 输入维度为 x_dim， 输出维度为 y_dim，那么参数个数 n 为：
n = 4 * ((x_dim + y_dim) * y_dim + y_dim)

'''

def bi_lstm_model():
    model = Sequential()
    model.add(Embedding(3800, 32, input_length=380))
    model.add(Dropout(0.5))
    model.add(Bidirectional(LSTM(32, return_sequences=True), merge_mode='concat'))
    # return_sequences=True表示每个时间步都输出
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    print(model.summary())


'''
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding_2 (Embedding)      (None, 380, 32)           121600    = 3800*32
_________________________________________________________________
dropout_3 (Dropout)          (None, 380, 32)           0         
_________________________________________________________________
bidirectional_1 (Bidirection (None, 380, 64)           16640     = (4 * ((32 + 64/2) * 64/2 + 64/2)) * 2
_________________________________________________________________
dense_3 (Dense)              (None, 380, 256)          16640     = 64*256+256
_________________________________________________________________
dropout_4 (Dropout)          (None, 380, 256)          0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 97280)             0         97280 = 380*256
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 97281     = 97280 + 1
=================================================================
Total params: 252,161
Trainable params: 252,161
Non-trainable params: 0
_________________________________________________________________
None

BiLSTM与LSTM相比就是多了个两倍的关系，
summary中的bidirectional_1输出为(None, 380, 64)，则一个单向LSTM的输出为(None, 380, 32)，即对应了训练参数中的units=32，
所以参数个数计算也就跟单向LSTM一样了，以上模型中的RNN层参数个数为
((32+32)*32*4+32*4)*2 = 16640

'''
def train(x_train, y_train, model):
    es = EarlyStopping(monitor='val_acc', patience=5)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train, y_train, validation_split=0.1, batch_size=64, epochs=20,
              callbacks=[es], shuffle=True)

def evaluate(x_test, y_test, model):
    scores = model.evaluate(x_test, y_test)

def main():
    lstm_model()
    bi_lstm_model()


if __name__ == '__main__':
    main()