#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 代码来源：https://www.jianshu.com/p/21b96d597367
# 数据集来源：https://archive.ics.uci.edu/ml/datasets/Beijing+PM2.5+Data

# print(keras.__version__, tf.__version__)
# 2.2.4-tf 1.14.0

# 归一化部分做了优化，训练和预测数据分别归一化，训练集做fit，然后对测试集进行transform

import pandas as pd
import numpy as np
from datetime import datetime
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import GRU, LSTM, Dense, Dropout
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from keras.metrics import binary_crossentropy, categorical_crossentropy
import os

'''
数据使用2010至2014年北京的空气污染数据，网上有很多下载的地方，可以自行搜索下载。
包括以下几个字段：'No', 'year', 'month', 'day', 'hour', 'pm2.5', 'DEWP', 'TEMP', 'PRES', 'cbwd', 'Iws', 'Is', 'Ir'。
其中pm2.5即为我们需要预测的字段，我设置的time_step =72。即，使用过去72小时的检测数据及污染数据，预测下一个小时的污染情况。

'''
# 格式化日期
def parse_date(year, month, day, hour):
    if len(str(month)) == 1:
        month = '0' + str(month)
    if len(str(day)) == 1:
        day = '0' + str(day)

    return datetime.strptime(str(year) + str(month) + str(day) + str(hour), '%Y%m%d%H').strftime(
        '%Y-%m-%d %H:00:00')  # 格式化日期


# 构建训练集、预测集，训练和预测分别transform
def make_train_test_data(path, ts, train_end, datasetsCount=None):
    air_data = pd.read_csv(path, header='infer')
    air_data['date'] = air_data.apply(lambda r: parse_date(r['year'], r['month'], r['day'], r['hour']), axis=1)  # 处理日期

    # 对风向cbwd进行encode
    encoder = LabelEncoder()
    air_data['cbwd'] = encoder.fit_transform(air_data['cbwd'])

    # 转化数据type
    air_data[['pm2.5', 'DEWP', 'TEMP', 'PRES', 'cbwd', 'Iws', 'Is', 'Ir']] = air_data[
        ['pm2.5', 'DEWP', 'TEMP', 'PRES', 'cbwd', 'Iws', 'Is', 'Ir']].astype('float64')
    air_data = air_data.dropna().reset_index(drop=True)

    # 使用过去N=ts小时数据预测未来一个小时的污染
    air_data['y'] = air_data['pm2.5']

    # 根据传入的参数缩减数据集
    if datasetsCount:
        air_data = air_data[:datasetsCount]

    # 切分训练和预测集
    train_all = air_data[:train_end]
    test_all = air_data[train_end:]

    # 训练数据进行归一化
    train_x = scalar_x.fit_transform(train_all[['pm2.5', 'DEWP', 'TEMP', 'PRES', 'cbwd', 'Iws', 'Is', 'Ir']])

    train_y = scalar_y.fit_transform(train_all[['y']])

    # 预测数据归一化
    test_x = scalar_x.transform(test_all[['pm2.5', 'DEWP', 'TEMP', 'PRES', 'cbwd', 'Iws', 'Is', 'Ir']])

    test_y = scalar_y.transform(test_all[['y']])

    # #############  构建训练和预测集  ###################
    ts_train_x = np.array([])
    ts_train_y = np.array([])

    ts_test_x = np.array([])
    ts_test_y = np.array([])

    # 构建训练数据集
    print('训练数据的原始shape：', train_x.shape)  # -> (8760, 8)
    for i in range(train_x.shape[0]):
        if i + ts == train_x.shape[0]:
            break

        ts_train_x = np.append(ts_train_x, train_x[i: i + ts, :])

        ts_train_y = np.append(ts_train_y, train_y[i + ts])

    # 构建预测数据集
    print('预测数据的原始shape：', test_x.shape) # (1240, 8)
    for i in range(test_x.shape[0]):
        if i + ts == test_x.shape[0]:
            break

        ts_test_x = np.append(ts_test_x, test_x[i: i + ts, :])

        ts_test_y = np.append(ts_test_y, test_y[i + ts])

    return ts_train_x.reshape((train_x.shape[0] - ts, ts, train_x.shape[1])), ts_train_y, \
           ts_test_x.reshape((test_x.shape[0] - ts, ts, test_x.shape[1])), ts_test_y, scalar_y

'''
训练及预测数据构建
将原来的一维数据转化为适合lstm的三维数据，shape=(batch_size, time_step, feature_dim)
思路（使用滑窗法）：
1、训练数据记为X，预测数据记为Y，行号记为i；
2、根据设置的time_step数，合并相应行数的X，转为一个(None, feature_dim*time_step)的二维数据X'；
3、根据time_step调整后的Y读取当前行号i+time_step后的index对应的Y的值；
4、对新的X'进行reshape，将其reshape为(sample_size, time_step, feature_dim)，该数据可直接进入model进行使用。

'''
# 构建model
def build_model(ts, fea_dim):
    model = Sequential()
    model.add(LSTM(64, input_shape=(ts, fea_dim), activation='sigmoid', return_sequences=True, dropout=0.01))
    model.add(LSTM(128, activation='sigmoid', return_sequences=True, dropout=0.01))
    model.add(Dropout(rate=0.01))
    model.add(LSTM(128, activation='sigmoid', dropout=0.01))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer=Adam(lr=0.002, decay=0.01))
    model.summary()
    return model

# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# lstm_4 (LSTM)                (None, 72, 64)            18688
# _________________________________________________________________
# lstm_5 (LSTM)                (None, 72, 128)           98816
# _________________________________________________________________
# dropout_2 (Dropout)          (None, 72, 128)           0
# _________________________________________________________________
# lstm_6 (LSTM)                (None, 128)               131584
# _________________________________________________________________
# dense_2 (Dense)              (None, 1)                 129
# =================================================================
# Total params: 249,217
# Trainable params: 249,217
# Non-trainable params: 0
# _________________________________________________________________
# Train on 6950 samples, validate on 1738 samples

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 超参设置
batch_size = 60
data_dim = 8
time_step = 72

# 归一化
scalar_x = MinMaxScaler(feature_range=(0, 1))
scalar_y = MinMaxScaler(feature_range=(0, 1))

# 获取训练和预测数据
# 使用365天的数据作为原始训练集
# 只使用10000小时的数据作为总数据
# 数据集来源于：https://archive.ics.uci.edu/ml/datasets/Beijing+PM2.5+Data
data_file_path = r"D:\\Users\\{}\\Downloads/PRSA_data_2010.1.1-2014.12.31.csv".format(os.getenv("USERNAME"))
x_train, y_train, x_test, y_test, scalar_Y = make_train_test_data(data_file_path, 72, 8760, 10000)
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
# (8688, 72, 8) (8688,) (1168, 72, 8) (1168,)

# 构建model
lstm_model = build_model(time_step, data_dim)

# 训练model，使用20%的数据作为验证集
lstm_model.fit(x_train, y_train, epochs=50, batch_size=60, validation_split=0.2)

# 预测结果
pred_y = lstm_model.predict(x_test)

# 转换为真实值
pred_y_inverse = scalar_Y.inverse_transform(pred_y)
true_y_inverse = scalar_Y.inverse_transform(y_test.reshape(len(y_test), 1))

# 归一化的y的mse
minmax_mse = mean_squared_error(y_pred=pred_y, y_true=y_test)

# 真实值的mse
true_mse = mean_squared_error(y_pred=pred_y_inverse, y_true=true_y_inverse)

print('归一化后的mse和真实mse分别是：', minmax_mse, true_mse)

# 画图观察
# 红线是预测曲线，黄色线是真实数据，从趋势看，拟合的还行，但从具体预测值上看，还有很大进步空间
fig = plt.figure(figsize=(20, 8))
plt.subplot(1, 1, 1)
plt.plot(pred_y_inverse, 'r', label='prediction')
plt.subplot(1, 1, 1)
plt.plot(true_y_inverse, 'y', label='true')
plt.legend(loc='best')
plt.show()

def main():
    pass


if __name__ == '__main__':
    main()