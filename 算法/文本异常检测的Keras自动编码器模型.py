#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from keras.models import Sequential, tf, Model
from keras.layers import Dense, Flatten, Embedding, Input, Bidirectional, Dropout, LSTM, GRU, TimeDistributed, RepeatVector
from keras.optimizers import adam_v2
from keras.losses import MSE, mean_squared_error
from keras import layers, activations, losses, metrics, optimizers
from keras.callbacks import EarlyStopping

import pickle as pkl

# X_train_pada_seq.shape
# (28840, 999)

input_length = 312

# https://www.5axxw.com/questions/content/6ymxhz

# 自动编码器是将输入 x 进行编码，得到新的特征 y ，并且希望原始的输入 x 能够从新的特征 y 重构出来。
# 我们希望自动编码器能够学习到在归一化转换时的特征，并且在应用时这个输入和输出是类似的。而对于异常情况，因模型没有学习此类数据特征，所以输入和输出将会有不同。
# 在对模型进行训练的过程中，只使用没有标签的正常数据训练。
# 这种方法的好处是它允许使用无监督的学习方式。
# 异常检测任务通常情况下负样本（异常样本）是比较少的，有时候依赖于人工标签，属于样本不平衡问题。
# 噪音。 异常和噪音有时候很难分清，位于数据的稀疏区域的点，与其他数据非常不同，因此可以断定为异常，但若周围有也有很多点分布，我们很难把点识别出来。

with open(os.path.join(os.path.split(EXCEL_DATA_FILE)[0], 'jd_api_train_dataset.pkl'), 'rb')as f:
    train_dataset = pkl.load(f)
    x_train = train_dataset['x_train']
    y_train = train_dataset['y_train']

# 只考虑多数类别的数据，少数类别的数据不纳入训练集；
x_test, y_test = [], []
X, Y = [], []

for x, y in zip(x_train, y_train):
    if y[0] == 1:
        x_test.append(x)
        y_test.append(y[0])
    else:
        X.append(x)
        Y.append(y[0])

X = np.array(X)
Y = np.array(Y)
x_test = np.array(x_test)
y_test = np.array(y_test)


import keras.backend as K

def reshape_squared_error(y_true, y_pred):
    '''自定义损失函数'''
    y_true = K.mean(y_true, axis=1)
    y_pred = K.mean(y_pred, axis=1)
    return mean_squared_error(y_true, y_pred)

def build_model(x_train, epochs=5):
    n_features = 312
    encoder = Sequential(name='encoder')
    encoder.add(layer=layers.Dense(units=20, activation=activations.relu, input_shape=x_train.shape[1:]))
    encoder.add(layers.Dropout(0.1))
    encoder.add(layer=layers.Dense(units=10, activation=activations.relu))
    encoder.add(layer=layers.Dense(units=n_features, activation=activations.relu))

    decoder = Sequential(name='decoder')
    decoder.add(layer=layers.Dense(units=10, activation=activations.relu, input_shape=x_train.shape[1:]))
    decoder.add(layer=layers.Dense(units=20, activation=activations.relu))
    decoder.add(layers.Dropout(0.1))
    decoder.add(layer=layers.Dense(units=n_features, activation=activations.sigmoid))

    autoencoder = Sequential([encoder, decoder])

    es = EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=20, restore_best_weights=True)
    autoencoder.compile(
    	loss=losses.MSE,
    	optimizer='Adam',
    	metrics=[metrics.mean_squared_error])
    autoencoder.summary()

    history = autoencoder.fit(
        x_train,
        x_train,
        epochs=epochs,
        batch_size=8,
        validation_split=0.1,
        callbacks = [es]
        # validation_data=(X_test_pada_seq, X_test_pada_seq)
    )
    return autoencoder

# autoencoder = build_model(X, epochs=20)

encoder_inputs = Input(shape=(x_train.shape[1], x_train.shape[2]))
# encoder_emb =  Embedding(input_dim=len(word_index)+1, output_dim=20, input_length=input_length)(encoder_inputs)
encoder_emb = encoder_inputs

encoder_LSTM_1 = Bidirectional(LSTM(32, activation='relu', return_sequences=True))(encoder_emb)
encoder_drop = Dropout(0.2)(encoder_LSTM_1)
encoder_LSTM_2 = Bidirectional(GRU(16, activation='relu', return_sequences=False,  name = 'bottleneck'))(encoder_drop)

decoder_repeated = RepeatVector(10)(encoder_LSTM_2)
decoder_LSTM = Bidirectional(LSTM(32, activation='relu', return_sequences=True))(decoder_repeated)
decoder_drop = Dropout(0.2)(decoder_LSTM)
# decoder_time = TimeDistributed(Dense(1, activation='softmax'))(decoder_drop)  # sigmoid
# decoder_output = tf.math.reduce_mean(decoder_time, axis=1)

decoder_output = TimeDistributed(Dense(X.shape[2], activation='softmax'))(decoder_drop)

autoencoder = Model(encoder_inputs, decoder_output)
# autoencoder.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
autoencoder.compile(loss=reshape_squared_error, optimizer='adam', metrics=['accuracy'])
autoencoder.summary()

history = autoencoder.fit(
                    X,
                    X,
                    epochs=5,
                    batch_size=8,
                    validation_split = 0.1,
                    # validation_data=(X_test_pada_seq, X_test_pada_seq)
                    )

# 使用该模型，我们能够计算出正常交易时的均方根误差，并且还能知道当需要均方根误差值为95%时，阈值应该设置为多少。
train_predicted_x = autoencoder.predict(x=X)
train_events_mse = mean_squared_error(X.mean(axis=1), train_predicted_x.mean(axis=1))
cut_off = np.percentile(train_events_mse, 95)


test_predicted_x = autoencoder.predict(x=x_test)
test_events_mse = mean_squared_error(x_test.mean(axis=1), test_predicted_x.mean(axis=1))

# 我们设置的阈值为 cut_off ，如果均方根误差大于cut_off 时，我们就把这次的交易视为异常交易，即有欺诈行为出现。
# 让我们选取size个异常数据和size个正常数据作为样本，结合阈值能够绘制如下图:

# 绘图
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 设置图形大小
plt.figure(figsize=(8, 4), dpi=80)
size = x_test.shape[0]
plt.plot(range(x_test.shape[0]), [t for t in train_events_mse[:size]], ls='-', lw=2, c='r', label='正常值')
plt.plot(range(x_test.shape[0]), [t for t in test_events_mse[:size]], ls='-.', lw=2, c='b', label='异常值')
plt.plot(range(x_test.shape[0]), [cut_off] * size, ls='-', lw=2, c='y', label='临界值')

plt.legend()
plt.xlabel('测试样本')  # 设置x轴的标签文本
plt.ylabel('mse')  # 设置y轴的标签文本
# plt.ylim(0, 5)  # 设置y轴展示范围
plt.title("正常值与异常值对比")
plt.show()


def main():
    pass


if __name__ == '__main__':
    main()