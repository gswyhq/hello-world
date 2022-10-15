#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# “孤立森林，Isolation Forest”，其思想是：假设我们用一个随机超平面来切割（split）数据空间（data space）, 切一次可以生成两个子空间（想象拿刀切蛋糕一分为二）。
# 之后我们再继续用一个随机超平面来切割每个子空间，循环下去，直到每子空间里面只有一个数据点为止。
# 直观上来讲，我们可以发现那些密度很高的簇是可以被切很多次才会停止切割，但是那些密度很低的点很容易很早的就停到一个子空间了。
#
# 注意：孤立森林不适用于特别高维的数据。由于每次切数据空间都是随机选取一个维度，建完树后仍然有大量的维度信息没有被使用，导致算法可靠性降低。
# 高维空间还可能存在大量噪音维度或无关维度（irrelevant attributes），影响树的构建。孤立森林算法具有线性时间复杂度。
# 因为是ensemble的方法，所以可以用在含有海量数据的数据集上面。
# 通常树的数量越多，算法越稳定。由于每棵树都是互相独立生成的，因此可以部署在大规模分布式系统上来加速运算。
#
# Isolation Forest 是无监督的异常检测算法，在实际应用时，并不需要黑白标签。
# 需要注意的是：（1）如果训练样本中异常样本的比例比较高，违背了先前提到的异常检测的基本假设，可能最终的效果会受影响；
# （2）异常检测跟具体的应用场景紧密相关，算法检测出的“异常”不一定是我们实际想要的。
# 比如，在识别虚假交易时，异常的交易未必就是虚假的交易。所以，在特征选择时，可能需要过滤不太相关的特征，以免识别出一些不太相关的“异常”。
#

import numpy as np
import matplotlib.pyplot as plt
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei'] #指定默认字体 
mpl.rcParams['axes.unicode_minus'] = False #解决保存图像是负号'-'显示为方块的问题
from sklearn.ensemble import IsolationForest


rng = np.random.RandomState(42)
# Generate train data
X = 0.3 * rng.randn(100, 2)
X_train = np.r_[X + 1, X - 3, X - 5, X + 6]
# Generate some regular novel observations
X = 0.3 * rng.randn(20, 2)
X_test = np.r_[X + 1, X - 3, X - 5, X + 6]
# Generate some abnormal novel observations
X_outliers = rng.uniform(low=-8, high=8, size=(20, 2))
# fit the model
clf = IsolationForest(max_samples=100*2, random_state=rng)
clf.fit(X_train)
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)
y_pred_outliers = clf.predict(X_outliers)
# plot the line, the samples, and the nearest vectors to the plane
xx, yy = np.meshgrid(np.linspace(-8, 8, 50), np.linspace(-8, 8, 50))
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.title("IsolationForest")
plt.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)
b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c='y')
b2 = plt.scatter(X_test[:, 0], X_test[:, 1], c='green')
c = plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='red')
plt.axis('tight')
plt.xlim((-8, 8))
plt.ylim((-8, 8))
plt.legend([b1, b2, c],
           ["训练",
            "正常测试", "异常测试"],
           loc="upper left")
plt.show()

def main():
    pass


if __name__ == '__main__':
    main()
