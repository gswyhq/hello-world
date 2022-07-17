#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import load_boston
from sklearn.model_selection import ShuffleSplit, KFold, cross_val_score
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold, StratifiedKFold, StratifiedShuffleSplit
from sklearn.datasets import load_breast_cancer

# 交叉验证，常用的就是KFold和 StratifiedKFold
# StratifiedKFold函数采用分层划分的方法（分层随机抽样思想），验证集中不同类别占比与原始样本的比例保持一致，故StratifiedKFold在做划分的时候需要传入标签特征。
# StratifiedKFold用法类似Kfold，但是他是分层采样，确保训练集，测试集中各类别样本的比例与原始数据集中相同。
# 1、KFold函数
# KFold函数共有三个参数：
# n_splits：默认为3，表示将数据划分为多少份，即k折交叉验证中的k；
# shuffle：默认为False，表示是否需要打乱顺序，这个参数在很多的函数中都会涉及，如果设置为True，则会先打乱顺序再做划分，如果为False，会直接按照顺序做划分；
# random_state：默认为None，表示随机数的种子，只有当shuffle设置为True的时候才会生效。

# Repeated K-Fold
# RepeatedKFold会重复KFold n次。
X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
random_state = 0
rkf = RepeatedKFold(n_splits=2, n_repeats=2, random_state=random_state)
for train, test in rkf.split(X):
   print("%s %s" % (train, test))

# 类似的，RepeatedStratifiedKFold也是重复Stratified K-Fold n次，每次用不同的随机数。

X = np.array([[1, 2], [3, 4], [1, 2], [3, 4],[5,9],[1,5],[3,9],[5,8],[1,1],[1,4]])
y = np.array([0, 1, 1, 1, 0, 0, 1, 0, 0, 0])

print('X:',X)
print('y:',y)

kf = KFold(n_splits=2 , shuffle=True, random_state=2020)

print(kf)
#做split时只需传入数据，不需要传入标签
for train_index, test_index in kf.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

X2 = np.array([[1, 2], [3, 4], [1, 2], [3, 4],[5,9],[1,5],[3,9],[5,8],[1,1],[1,4]])
y = np.array([0, 1, 1, 1, 0, 0, 1, 0, 0, 0])

print('X2:',X2)
print('y:',y)

# StratifiedKFold
skf = StratifiedKFold(n_splits=2, random_state=2020, shuffle=True)
print(skf)

#做划分是需要同时传入数据集和标签
for train_index, test_index in skf.split(X2, y):
    print('TRAIN:', train_index, "TEST:", test_index)
    X_train, X_test = X2[train_index], X2[test_index]
    y_train, y_test = y[train_index], y[test_index]


# 随机排列交叉验证
from sklearn.model_selection import ShuffleSplit
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
y = np.array([1, 2, 1, 2])
rs = ShuffleSplit(n_splits=3, test_size=.25, random_state=0)
rs.get_n_splits(X)

for train_index, test_index in rs.split(X):
       print("TRAIN:", train_index, "TEST:", test_index)

# 分层k折
# StratifiedKFold 每个小集合中， 各个类别的样例比例大致和完整数据集中相同。
from sklearn.model_selection import StratifiedKFold
X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
y = np.array([0, 0, 1, 1])
skf = StratifiedKFold(n_splits=2)
skf.get_n_splits(X, y)
2
for train_index, test_index in skf.split(X, y):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

# 分层随机split
# StratifiedShuffleSplit 创建一个划分，但是划分中每个类的比例和完整数据集中的相同。

from sklearn.model_selection import StratifiedShuffleSplit
X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
y = np.array([0, 0, 1, 1])
sss = StratifiedShuffleSplit(n_splits=3, test_size=0.5, random_state=0)
sss.get_n_splits(X, y)
# 3
for train_index, test_index in sss.split(X, y):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

def ada_boost(X, Y, **kwargs):
    clf = AdaBoostClassifier(n_estimators=50)  # 指定50个弱分类器
    # cv = ShuffleSplit(n_splits=10, test_size=0.3, random_state=0)
    cv = KFold(n_splits=10, shuffle=True, random_state=42)
    scores = cross_val_score(clf, np.array(X), np.array(Y), cv=cv, scoring='average_precision')  # 调用模型，形成n_splits个评分结果；
    # 模型 数据集 目标变量；
    # 交叉验证（cross-validation，简称CV）, k折交叉验证；设置n_splits=10，也就是会分割为10分子集，然后去遍历调用模型, 若不设置，默认是输出5个结果；
    # scoring 参数设定评分函数，默认是准确率；可选值可参考：sorted(sklearn.metrics.SCORERS.keys())
    print(scores.mean())

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# 加载数据
iris = load_iris()

# 划分数据集
x_train,x_test,y_train,y_test = train_test_split(iris.data,iris.target,test_size=0.3,random_state=8)

# 标准化
transfer = StandardScaler()
x_train = transfer.fit_transform(x_train)
x_test = transfer.transform(x_test)

# 指定算法及模型选择与调优——网格搜索和交叉验证
estimator = KNeighborsClassifier()
param_dict = {"n_neighbors": [1, 3, 5]}
estimator = GridSearchCV(estimator, param_grid=param_dict, cv=3)

# 训练模型
estimator.fit(x_train,y_train)

# 模型评估
# 方法一 比对真实值与预测值
y_predict = estimator.predict(x_test)
y_test == y_predict
# 方法二 计算准确率
estimator.score(x_test,y_test)

# 然后进行评估查看最终选择的结果和交叉验证的结果
print("在交叉验证中验证的最好结果：\n", estimator.best_score_)
print("最好的参数模型：\n", estimator.best_estimator_)
print("每次交叉验证后的准确率结果：\n", estimator.cv_results_)


# 使用了kfold之后，是形成了k个model吗？如果model的表现效果是用k个评分的平均值来表示，那最终应该保存哪个model呢？
# 其实交叉验证的根本作用是：在数据量较少时，用来选择哪个【model+超参】
# （这里注意区分超参数和模型参数的不同，超参是人为设定的，比如learning rate学习率，而参数是需要等模型见到data后学习训练得到的，比如linear regression中的特征权重）
# 因为当数据量很少的时候，train和test的划分不同其实对模型的训练有很大的影响，为了避免这种扰动，就需要取多组不同的train和test，
# 这样做还有一个好处，就是可以让模型学到所有数据上的信息，而不会总有一部分会被划分出去作为test
# 故而，交叉验证会形成多个模型，但我们最后保存的模型并不是从这里头选择；而是根据交叉验证，确定好最优的模型及其超参数；
# 确定好了模型及其超参数，再用全部数据训练一遍。

def main():
    data = load_breast_cancer()
    X = data.data
    Y = data.target
    ada_boost(X, Y)


if __name__ == '__main__':
    main()