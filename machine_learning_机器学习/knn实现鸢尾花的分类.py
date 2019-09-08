#!/usr/bin/python3
# coding: utf-8

from sklearn import datasets
# 导入内置数据集模块                      
from sklearn.neighbors import KNeighborsClassifier
# 导入sklearn.neighbors模块中KNN类
import numpy as np

np.random.seed(0)
# 设置随机种子，不设置的话默认是按系统时间作为参数，因此每次调用随机模块时产生的随机数都不一样设置后每次产生的一样
iris = datasets.load_iris()
# 导入鸢尾花的数据集，iris是一个类似于结构体的东西，内部有样本数据，如果是监督学习还有标签数据
iris_x = iris.data
# 样本数据150*4二维数据，代表150个样本，每个样本4个属性分别为花瓣和花萼的长、宽
iris_y = iris.target
# 长150的以为数组，样本数据的标签
indices = np.random.permutation(len(iris_x))
# permutation接收一个数作为参数(150),产生一个0-149一维数组，只不过是随机打乱的，当然她也可以接收一个一维数组作为参数，结果是直接对这个数组打乱
iris_x_train = iris_x[indices[:-10]]
# 随机选取140个样本作为训练数据集
iris_y_train = iris_y[indices[:-10]]
# 并且选取这140个样本的标签作为训练数据集的标签
iris_x_test = iris_x[indices[-10:]]
# 剩下的10个样本作为测试数据集
iris_y_test = iris_y[indices[-10:]]
# 并且把剩下10个样本对应标签作为测试数据及的标签

knn = KNeighborsClassifier(n_neighbors=5,
                 weights='uniform', algorithm='auto', leaf_size=30,
                 p=2, metric='minkowski', metric_params=None, n_jobs=None)


# n_neighbors = 5
# int 型参数
# knn算法中指定以最近的几个最近邻样本具有投票权，默认参数为5

# weights = 'uniform'
# str参数
# 即每个拥有投票权的样本是按什么比重投票，'uniform'
# 表示等比重投票，'distance'表示按距离反比投票，
# [callable]表示自己定义的一个函数，这个函数接收一个距离数组，返回一个权值数组。默认参数为‘uniform’

# algrithm = 'auto'
# str参数,即内部采用什么算法实现。有以下几种选择参数：'ball_tree': 球树、'kd_tree': kd树、'brute': 暴力搜索、'auto': 自动根据数据的类型和结构选择合适的算法。
# 默认情况下是‘auto’。暴力搜索就不用说了大家都知道。具体前两种树型数据结构哪种好视情况而定。
# KD树是对依次对K维坐标轴，以中值切分构造的树, 每一个节点是一个超矩形，在维数小于20时效率最高 。
# ball_tree
# 是为了克服KD树高维失效而发明的，其构造过程是以质心C和半径r分割样本空间，每一个节点是一个超球体。一般低维数据用kd_tree速度快，用ball_tree相对较慢。
# 超过20维之后的高维数据用kd_tree效果反而不佳，而ball_tree效果要好。

# leaf_size = 30
# int参数
# 基于以上介绍的算法，此参数给出了kd_tree或者ball_tree叶节点规模，叶节点的不同规模会影响数的构造和搜索速度，同样会影响储树的内存的大小。

# matric = 'minkowski'
# str或者距离度量对象
# 即怎样度量距离。默认是闵氏距离，闵氏距离不是一种具体的距离度量方法，它可以说包括了其他距离度量方式，是其他距离度量的推广，
# 具体各种距离度量只是参数p的取值不同或者是否去极限的不同情况

# p = 2
# int参数
# 就是以上闵氏距离各种不同的距离参数，默认为2，即欧氏距离。p = 1
# 代表曼哈顿距离等等

# metric_params = None
# 距离度量函数的额外关键字参数，一般不用管，默认为None
#
# n_jobs = 1
# int参数
# 指并行计算的线程数量，默认为1表示一个线程，为 - 1
# 的话表示为CPU的内核数，也可以指定为其他数量的线程，这里不是很追求速度的话不用管，需要用到的话去看看多线程。

# 定义一个knn分类器对象
knn.fit(iris_x_train, iris_y_train)
# 调用该对象的训练方法，主要接收两个参数：训练数据集及其样本标签

iris_y_predict = knn.predict(iris_x_test)
# 调用该对象的测试方法，主要接收一个参数：测试数据集
probility = knn.predict_proba(iris_x_test)
# 计算各测试样本基于概率的预测
neighborpoint = knn.kneighbors([iris_x_test[-1]], 5, False)
# 计算与最后一个测试样本距离在最近的5个点，返回的是这些样本的序号组成的数组
score = knn.score(iris_x_test, iris_y_test, sample_weight=None)
# 调用该对象的打分方法，计算出准确率

print('测试的结果 = ')
print(iris_y_predict)
# 输出测试的结果

print('原始测试数据集的正确标签 = ')
print(iris_y_test)
# 输出原始测试数据集的正确标签，以方便对比
print('Accuracy:', score)
# 输出准确率计算结果
print('与最后一个测试样本距离在最近的5个点，返回的是这些样本的序号组成的数组:', neighborpoint)

print('各测试样本基于概率的预测:', probility)

def main():
    pass


if __name__ == '__main__':
    main()