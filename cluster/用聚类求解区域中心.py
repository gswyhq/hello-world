#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.cluster import KMeans, DBSCAN, MiniBatchKMeans, MeanShift
from sklearn.datasets import load_iris
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from scipy.spatial.distance import pdist, squareform

from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei'] #指定默认字体   
mpl.rcParams['axes.unicode_minus'] = False #解决保存图像是负号'-'显示为方块的问题

# 添加一些噪声，增加一些不确定性。定义噪声函数：
def add_noise(x, y, amplitude):
    X = np.concatenate((x, y))
    X += amplitude * np.random.randn(2, X.shape[1])
    return X.T

# 为了演示凝聚层次聚类的优势，我们用它对一些在空间中是连接在一起、但彼此却非常接近的数据进行聚类。我们希望连接在一起的数据可以聚成一类，而不是在空间上非常接近的点聚成一类。下面定义一个函数来获取一组呈螺旋状的数据点：
def get_spiral(t, noise_amplitude=0.5):
    r = t
    x = r * np.cos(t)
    y = r * np.sin(t)

    return add_noise(x, y, noise_amplitude)


def test3():

    # Load the data
    iris = load_iris()
    X, y = iris.data, iris.target

    # shuffle the data
    shuffle = np.random.permutation(np.arange(X.shape[0]))
    X = X[shuffle]

    # scale X
    X = (X - X.mean()) / X.std()

    # plot K-means centroids
    km = KMeans(n_clusters=1, n_init=10)  # establish the model

    # fit the data
    km.fit(X)

    # km centers
    print(km.cluster_centers_)

def test4():
    n_samples = 500
    np.random.seed(2)
    t = 2.5 * np.pi * (1 + 2 * np.random.rand(1, n_samples))
    X = get_spiral(t)
    X = [[10+10*np.random.random(), 10+10*np.random.random()] for _ in range(20)] + [[100+20*np.random.random(), 50+10*np.random.random()] for _ in range(30)]
    X = np.array(X)

    # X, y = make_blobs(n_samples=900, n_features=2, centers=3, cluster_std=[1.0, 1.0, 1.0], random_state=100)
    print(X)
    # plot K-means centroids
    # km = KMeans(n_clusters=1, n_init=10)  # establish the model
    # km = MiniBatchKMeans(n_clusters=1)
    km = MeanShift()
    # DBSCAN, MiniBatchKMeans, MeanShift
    # fit the data
    km.fit(X)

    # km centers
    print(km.cluster_centers_)

    cluster_center_x = km.cluster_centers_[0][0]
    cluster_center_y = km.cluster_centers_[0][1]
    plt.figure()
    # specify marker shapes for different clusters
    markers = '.v*'
    plt.scatter([x for x, y in X], [y for x, y in X], s=50,
                    marker=markers[0], color='c', facecolors='r')
    plt.scatter([cluster_center_x], [cluster_center_y], s=50,
                marker=markers[1], color='y', facecolors='r')
    center_sorted = sorted(list(X), key=lambda x: math.pow(math.fabs(x[0]-cluster_center_x), 2)+math.pow(math.fabs(x[1]-cluster_center_y), 2))
    center_x = center_sorted[0][0]
    center_y = center_sorted[0][1]


    plt.scatter([center_x], [center_y], s=50,
                marker=markers[2], color='b', facecolors='r')
    print(center_sorted[0])
    plt.show()


def main():
    pass


if __name__ == '__main__':
    # main()
    # test2()
    # test3()
    test4()


