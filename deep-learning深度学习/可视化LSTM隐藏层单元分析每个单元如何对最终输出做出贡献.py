#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 可视化LSTM隐藏层单元，分析每个单元如何对最终输出做出贡献
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import re

# 可视化库
from IPython.display import HTML as html_print
from IPython.display import display
import keras.backend as K

from keras.models import load_model


# Read data
filename = "./result/wonderland.txt"  # https://github.com/Praneet9/Visualising-LSTM-Activations/blob/master/wonderland.txt
raw_text = open(filename, 'r', encoding='utf-8').read()
raw_text = re.sub(r'[ ]+', ' ', raw_text)

# # 创建字符到整数的映射(create mapping of unique chars to integers)
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))

n_chars = len(raw_text)
n_vocab = len(chars)
print("Total Characters: ", n_chars)
print("Total Vocab: ", n_vocab)
# Total
# Characters: 143540
# Total
# Vocab: 73

# prepare the dataset of input to output pairs encoded as integers
seq_length = 100
dataX = []
dataY = []

for i in range(0, n_chars - seq_length, 1):
    seq_in = raw_text[i:i + seq_length]
    seq_out = raw_text[i + seq_length]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])

n_patterns = len(dataX)
print("Total Patterns: ", n_patterns)
# Total
# Patterns: 143440

# reshape X to be [samples, time steps, features]
X = np.reshape(dataX, (n_patterns, seq_length, 1))

# 标准化(normalize)
X = X / float(n_vocab)

# # one-hot编码(one hot encode the output variable)
y = np_utils.to_categorical(dataY)

# define the checkpoint
filepath = "./result/weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

# define the LSTM model
model = Sequential()

model.add(LSTM(64, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.5))

model.add(LSTM(64))
model.add(Dropout(0.5))

model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# lstm (LSTM)                  (None, 100, 64)           16896
# _________________________________________________________________
# dropout (Dropout)            (None, 100, 64)           0
# _________________________________________________________________
# lstm_1 (LSTM)                (None, 64)                33024
# _________________________________________________________________
# dropout_1 (Dropout)          (None, 64)                0
# _________________________________________________________________
# dense (Dense)                (None, 73)                4745
# =================================================================
# Total params: 54,665
# Trainable params: 54,665
# Non-trainable params: 0
# _________________________________________________________________

model.fit(X, y, epochs=1, batch_size=16, callbacks=callbacks_list)
# 8965/8965 [==============================] - 802s 89ms/step - loss: 3.0354 - accuracy: 0.1920
#
# Epoch 00001: loss improved from inf to 3.03540, saving model to ./result/weights-improvement-01-3.0354.hdf5

# 加载训练好的模型
filename = "./result/weights-improvement-01-3.0354.hdf5"
model2 = load_model(filename)
model2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# #从中间层获取输出以可视化激活
lstm = model2.layers[2]  # #第三层是输出形状为LSTM层,这里是可视化LSTM层激活
attn_func = K.function(inputs=[model2.input],
                       outputs=[lstm.output]
                       )

# get html element
def cstr(s, color='black'):
    '''获取HTML元素'''
    if s == ' ':
        return "<text style=color:#000;padding-left:10px;background-color:{}> </text>".format(color, s)
    else:
        return "<text style=color:#000;background-color:{}>{} </text>".format(color, s)


# print html
def print_color(t):
    display(html_print(''.join([cstr(ti, color=ci) for ti, ci in t])))


# get appropriate color for value
def get_clr(value):
    '''需要一个可以表示其对整个输出重要性的规模值。get_clr功能有助于获得给定值的适当颜色。
    '''
    # 定义显示颜色梯度
    # 将Sigmoid应用于图层输出后，值在0到1的范围内。数字越接近1，它的重要性就越高。
    #     如果该数字接近于0，则意味着不会以任何主要方式对最终预测做出贡献。
    #     这些单元格的重要性由颜色表示，其中蓝色表示较低的重要性，红色表示较高的重要性。
    colors = ['#85c2e1', '#89c4e2', '#95cae5', '#99cce6', '#a1d0e8',
              '#b2d9ec', '#baddee', '#c2e1f0', '#eff7fb', '#f9e8e8',
              '#f9e8e8', '#f9d4d4', '#f9bdbd', '#f8a8a8', '#f68f8f',
              '#f47676', '#f45f5f', '#f34343', '#f33b3b', '#f42e2e']
    value = int((value * 100) / 5)
    return colors[value]

# sigmoid function
def sigmoid(x):
    z = 1 / (1 + np.exp(-x))
    return z

def visualize(output_values, result_list, cell_no):
    '''
    visualize函数将预测序列，序列中每个字符的通过sigmoid函数传递激活以及要可视化的单元格编号作为输入。
    根据输出的值，将以适当的背景色打印字符。
    output_values：LSTM层激活
    result_list： 预测的字符
    '''
    print("\nCell Number:", cell_no, "\n")
    text_colours = []
    for i in range(len(output_values)):
        # 遍历每个字符位置，获取每个字符及其该字符在对应单元格sigmoid激活对应重要性颜色
        text = (result_list[i], get_clr(output_values[i][cell_no]))
        text_colours.append(text)
    print_color(text_colours)

# 从随机序列中获得预测(Get Predictions from random sequence)
def get_predictions(data):
    '''get_predictions函数随机选择一个输入种子序列，并获得该种子序列的预测序列。
    visualize函数将预测序列，序列中每个字符的sigmoid函数传递激活以及要可视化的单元格编号作为输入。
    根据输出的值，将以适当的背景色打印字符。
    将Sigmoid应用于图层输出后，值在0到1的范围内。数字越接近1，它的重要性就越高。
    如果该数字接近于0，则意味着不会以任何主要方式对最终预测做出贡献。
    这些单元格的重要性由颜色表示，其中蓝色表示较低的重要性，红色表示较高的重要性。
    return：
    output_values：LSTM层激活
    result_list： 预测的字符
    '''
    start = np.random.randint(0, len(data) - 1)
    pattern = data[start]
    result_list, output_values = [], []
    print("Seed:")
    print("\"" + ''.join([int_to_char[value] for value in pattern]) + "\"")
    print("\nGenerated:")

    # 预测100个字符；
    for i in range(100):
        # Reshaping input array for predicting next character
        x = np.reshape(pattern, (1, len(pattern), 1))
        x = x / float(n_vocab)

        # Prediction
        prediction = model2.predict(x, verbose=0)

        # LSTM Activations LSTM层的激活
        output = attn_func([x])[0][0]
        output = sigmoid(output)
        output_values.append(output)

        # 预测字符(Predicted Character)
        index = np.argmax(prediction)
        result = int_to_char[index]

        # # 为下一个字符准备输入(Preparing input for next character)
        seq_in = [int_to_char[value] for value in pattern]
        pattern.append(index)
        pattern = pattern[1:len(pattern)]

        # 保存预测的字符(Saving generated characters)
        result_list.append(result)
    return output_values, result_list

# 挑选一些可以理解的单元可视化
output_values, result_list = get_predictions(dataX)
# 可能大多数LSTM单元未显示任何可理解的模式。可以可视化了所有LSTM隐藏层单元，挑选一些可以理解的单元可视化。
for cell_no in [9, 35, 63]:
    visualize(output_values, result_list, cell_no)


def main():
    pass


if __name__ == '__main__':
    main()