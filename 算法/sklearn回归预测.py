#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import numpy as np
import matplotlib.pyplot as plt

from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体   
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

###########1.数据生成部分##########
def f(x1):
    y = 0.5 * np.sin(x1) + 3 + 0.1 * x1
    return y

def load_data():
    x1_train = np.linspace(0,50,500)
    data_train = np.array([[x1,f(x1) + (np.random.random(1)-0.5)] for x1 in x1_train])
    x1_test = np.linspace(0,50,100)+ 0.5 * np.random.random(100)
    data_test = np.array([[x1,f(x1)] for x1 in x1_test])
    x1_future = np.linspace(50,100,100)+ 0.5 * np.random.random(100)
    data_future = np.array([[x1,f(x1)] for x1 in x1_future])
    return data_train, data_test, data_future

train, test, future = load_data()
x_train, y_train = train[:,:1], train[:,1] #数据前两列是x1,x2 第三列是y,这里的y有随机噪声
x_test ,y_test = test[:,:1], test[:,1] # 同上,不过这里的y没有噪声
x_future, y_future = future[:,:1] ,future[:,1] # 同上,不过这里的y没有噪声


plt.figure(figsize=(15,5))
plt.plot(x_train[:,0],y_train,label='train')
plt.plot(x_test[:,0],y_test,label='test')
plt.plot(x_future[:,0],y_future,label='future')
plt.legend()
plt.show()


# sklearn的使用
# 实例化一个算法对象
# 调用fit()函数
# 使用predict()函数进行预测
# 使用score()函数来评估预测值和真实值的使用
###########2.回归部分##########
def try_different_method(model, title=''):
    model.fit(x_train,y_train)
    score = model.score(x_test, y_test)
    result = model.predict(x_test)
    future = model.predict(x_future)
    plt.figure(figsize=(15,5))
    plt.plot(np.arange(len(result)), y_test,'go-',label='true value')
    plt.plot(np.arange(len(result)),result,'ro-',label='predict value')
    plt.plot(np.arange(len(result),len(result)+len(future)),y_future,'yo-',label='future true value')
    plt.plot(np.arange(len(result),len(result)+len(future)),future,'bo-',label='future predict value')
    plt.title('{} score: {:.4f}'.format(title, score))
    plt.legend()
    plt.show()

###########3.具体方法选择##########
####3.1决策树回归####
from sklearn import tree
model_DecisionTreeRegressor = tree.DecisionTreeRegressor()
####3.2线性回归####
from sklearn import linear_model
model_LinearRegression = linear_model.LinearRegression()
####3.3SVM回归####
from sklearn import svm
model_SVR = svm.SVR()
####3.4KNN回归####
from sklearn import neighbors
model_KNeighborsRegressor = neighbors.KNeighborsRegressor()
####3.5随机森林回归####
from sklearn import ensemble
model_RandomForestRegressor = ensemble.RandomForestRegressor(n_estimators=20)#这里使用20个决策树
####3.6Adaboost回归####
from sklearn import ensemble
model_AdaBoostRegressor = ensemble.AdaBoostRegressor(n_estimators=50)#这里使用50个决策树
####3.7GBRT回归####
from sklearn import ensemble
model_GradientBoostingRegressor = ensemble.GradientBoostingRegressor(n_estimators=100)#这里使用100个决策树
####3.8Bagging回归####
from sklearn.ensemble import BaggingRegressor
model_BaggingRegressor = BaggingRegressor()
####3.9ExtraTree极端随机树回归####
from sklearn.tree import ExtraTreeRegressor
model_ExtraTreeRegressor = ExtraTreeRegressor()
####3.10ARD贝叶斯ARD回归
model_ARDRegression = linear_model.ARDRegression()
####3.11BayesianRidge贝叶斯岭回归
model_BayesianRidge = linear_model.BayesianRidge()
####3.12TheilSen泰尔森估算
model_TheilSenRegressor = linear_model.TheilSenRegressor()
####3.13RANSAC随机抽样一致性算法
model_RANSACRegressor = linear_model.RANSACRegressor()



# 结果展示
# 决策树回归结果

###########4.具体方法调用部分##########
try_different_method(model_DecisionTreeRegressor, title='决策树')


# 线性回归结果
try_different_method(model_LinearRegression, title='线性回归')


# SVM回归结果
try_different_method(model_SVR, title='SVM回归')


# KNN回归结果
try_different_method(model_KNeighborsRegressor, title='knn回归')


# 随机森林回归结果
try_different_method(model_RandomForestRegressor, title='随机森林回归')


# Adaboost回归结果
try_different_method(model_AdaBoostRegressor, title='Adaboost回归')


# GBRT回归结果
try_different_method(model_GradientBoostingRegressor, title='GBRT回归')


# Bagging回归结果
try_different_method(model_BaggingRegressor, title='Bagging回归')


# 极端随机树回归结果
try_different_method(model_ExtraTreeRegressor, title='极端随机树回归')


# 贝叶斯ARD回归结果
try_different_method(model_ARDRegression, title='贝叶斯ARD回归')


# 贝叶斯岭回归结果
try_different_method(model_BayesianRidge, title='贝叶斯岭回归')


# 泰尔森估算回归结果
try_different_method(model_TheilSenRegressor, title='泰尔森估算回归')


# 随机抽样一致性算法
try_different_method(model_RANSACRegressor, title='随机抽样一致性算法')


def main():
    pass


if __name__ == '__main__':
    main()