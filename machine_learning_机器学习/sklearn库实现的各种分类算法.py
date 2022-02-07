#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# KNN
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
def KNN(X,y,XX):#X,y 分别为训练数据集的数据和标签，XX为测试数据
  model = KNeighborsClassifier(n_neighbors=10)#默认为5
  model.fit(X,y)
  predicted = model.predict(XX)
  return predicted

# SVM
from sklearn.svm import SVC
def SVM(X,y,XX):
  model = SVC(c=5.0)
  model.fit(X,y)
  predicted = model.predict(XX)
  return predicted

# SVM Classifier using cross validation
def svm_cross_validation(train_x, train_y):
  from sklearn.grid_search import GridSearchCV
  from sklearn.svm import SVC
  model = SVC(kernel='rbf', probability=True)
  param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], 'gamma': [0.001, 0.0001]}
  grid_search = GridSearchCV(model, param_grid, n_jobs = 1, verbose=1)
  grid_search.fit(train_x, train_y)
  best_parameters = grid_search.best_estimator_.get_params()
  for para, val in list(best_parameters.items()):
    print(para, val)
  model = SVC(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'], probability=True)
  model.fit(train_x, train_y)
  return model

# LR
from sklearn.linear_model import LogisticRegression
def LR(X,y, XX):
  model = LogisticRegression()
  model.fit(X,y)
  predicted = model.predict(XX)
  return predicted

# 决策树（CART）
from sklearn.tree import DecisionTreeClassifier
def CTRA(X,y,XX):
  model = DecisionTreeClassifier()
  model.fit(X,y)
  predicted = model.predict(XX)
  return predicted

# 随机森林
from sklearn.ensemble import RandomForestClassifier
def RFC(X,y,XX):
  model = RandomForestClassifier()
  model.fit(X,y)
  predicted = model.predict(XX)
  return predicted

# GBDT(Gradient Boosting Decision Tree)
from sklearn.ensemble import GradientBoostingClassifier
def GBC(X,y,XX):
  model = GradientBoostingClassifier()
  model.fit(X,y)
  predicted = model.predict(XX)
  return predicted

# 朴素贝叶斯：一个是基于高斯分布求概率，一个是基于多项式分布求概率，一个是基于伯努利分布求概率。
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
def GNB(X,y,XX):
  model =GaussianNB()
  model.fit(X,y)
  predicted = model.predict(XX)
  return predicted
def MNB(X,y,XX):
  model = MultinomialNB()
  model.fit(X,y)
  predicted = model.predict(XX)
  return predicted
def BNB(X,y,XX):
  model = BernoulliNB()
  model.fit(X,y)
  predicted = model.predict(XX)
  return predicted


def main():
    pass


if __name__ == '__main__':
    main()