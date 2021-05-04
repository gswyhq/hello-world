#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import numpy as np
np.random.seed(1337)
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

# 生成200组数据
X = np.linspace(-1, 1, 200) #在返回（-1, 1）范围内的等差序列
np.random.shuffle(X)    # 打乱顺序
Y = 0.5 * X + 2 + np.random.normal(0, 0.05, (200, )) #生成Y并添加噪声
# plot
plt.scatter(X, Y)
plt.show()

X_train, Y_train = X[:160], Y[:160]     # 前160组数据为训练数据集
X_test, Y_test = X[160:], Y[160:]      #后40组数据为测试数据集

# 构建神经网络模型
model = Sequential()
model.add(Dense(20,input_shape=(1,)))#第一隐藏层20个节点
model.add(Dense(1))#第二层1个节点

# 选定loss函数和优化器
model.compile(loss='mse', optimizer='sgd')

# 训练过程
print('Training -----------')
for step in range(501):
    cost = model.train_on_batch(X_train, Y_train)
    if step % 50 == 0:
        print("After %d trainings, the cost: %f" % (step, cost))

# 测试过程
print('\nTesting ------------')
cost = model.evaluate(X_test, Y_test, batch_size=40)
print('test cost:', cost)
W1, b1 = model.layers[0].get_weights()
print('Weights=', W1, '\nbiases=', b1)

W2, b2 = model.layers[1].get_weights()
print('Weights=', W2, '\nbiases=', b2)

# 将训练结果绘出
Y_pred = model.predict(X_test)
plt.scatter(X_test, Y_test)
plt.plot(X_test, Y_pred)
plt.show()

def main():
    pass


if __name__ == '__main__':
    main()