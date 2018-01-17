#!/usr/bin/python3
# coding: utf-8

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

X = [[6, 2], [8, 1], [10, 0], [14, 2], [18, 0]]
y = [[7], [9], [13], [17.5], [18]]

# 多元线性回归：
model = LinearRegression()
model.fit(X, y)
X_test = [[8, 2], [9, 0], [11, 2], [16, 2], [12, 0]]
y_test = [[11], [8.5], [15], [18], [11]]
predictions = model.predict(X_test)
for i, prediction in enumerate(predictions):
    print('Predicted: %s, Target: %s' % (prediction, y_test[i]))
print('R-squared: %.2f' % model.score(X_test, y_test))

# 多元多项式回归
X_train = [[6, 2], [8, 1], [10, 0], [14, 2], [18, 0]]
y_train = [[7], [9], [13], [17.5], [18]]

quadratic_featurizer = PolynomialFeatures(degree=2)
X_train_quadratic = quadratic_featurizer.fit_transform(X_train)

regressor_quadratic = LinearRegression()
# 训练
regressor_quadratic.fit(X_train_quadratic, y_train)

xx_quadratic = quadratic_featurizer.transform([[6, 3]])
print("预测结果： {}".format(regressor_quadratic.predict(xx_quadratic)))


def main():
    pass


if __name__ == '__main__':
    main()