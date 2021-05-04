#!/usr/bin/python3
# coding: utf-8

import time
import numpy as np
import copy
import pickle
from sklearn.datasets import make_moons
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class LVQ():
    def __init__(self, max_iter=10000, eta=0.1, e=0.01):
        self.max_iter = max_iter
        self.eta = eta
        self.e = e

    def dist(self, x1, x2):
        return np.linalg.norm(x1 - x2)

    def get_mu(self, X, Y):
        k = len(set(Y))
        index = np.random.choice(X.shape[0], 1, replace=False)
        mus = []
        mus.append(X[index])
        mus_label = []
        mus_label.append(Y[index])
        for _ in range(k - 1):
            max_dist_index = 0
            max_distance = 0
            for j in range(X.shape[0]):
                min_dist_with_mu = 999999

                for mu in mus:
                    dist_with_mu = self.dist(mu, X[j])
                    if min_dist_with_mu > dist_with_mu:
                        min_dist_with_mu = dist_with_mu

                if max_distance < min_dist_with_mu:
                    max_distance = min_dist_with_mu
                    max_dist_index = j
            mus.append(X[max_dist_index])
            mus_label.append(Y[max_dist_index])

        mus_array = np.array([])
        for i in range(k):
            if i == 0:
                mus_array = mus[i]
            else:
                mus[i] = mus[i].reshape(mus[0].shape)
                mus_array = np.append(mus_array, mus[i], axis=0)
        mus_label_array = np.array(mus_label)
        return mus_array, mus_label_array

    def get_mu_index(self, x, mus_array=None):
        if not mus_array is None:
            self.mus_array = mus_array
        min_dist_with_mu = 999999
        index = -1

        for i in range(self.mus_array.shape[0]):
            dist_with_mu = self.dist(self.mus_array[i], x)
            if min_dist_with_mu > dist_with_mu:
                min_dist_with_mu = dist_with_mu
                index = i

        return index

    def fit(self, X, Y):
        self.mus_array, self.mus_label_array = self.get_mu(X, Y)
        iter = 0

        while(iter < self.max_iter):
            old_mus_array = copy.deepcopy(self.mus_array)
            index = np.random.choice(Y.shape[0], 1, replace=False)

            mu_index = self.get_mu_index(X[index])
            if self.mus_label_array[mu_index] == Y[index]:
                self.mus_array[mu_index] = self.mus_array[mu_index] + \
                    self.eta * (X[index] - self.mus_array[mu_index])
            else:
                self.mus_array[mu_index] = self.mus_array[mu_index] - \
                    self.eta * (X[index] - self.mus_array[mu_index])

            diff = 0
            for i in range(self.mus_array.shape[0]):
                diff += np.linalg.norm(self.mus_array[i] - old_mus_array[i])
            if diff < self.e:
                print('迭代{}次退出'.format(iter))
                return
            iter += 1
        print("迭代超过{}次，退出迭代".format(self.max_iter))

    def init_animate(self, X, Y, X2, Y2):
        self.mus_array1, self.mus_label_array1 = self.get_mu(X, Y)
        self.mus_array2, self.mus_label_array2 = self.get_mu(X2, Y2)
        self.X = X
        self.Y = Y
        self.X2 = X2
        self.Y2 = Y2

    def animate(self, i):
        print('迭代次数：{}'.format(i))
        old_mus_array = copy.deepcopy(self.mus_array1)
        index = np.random.choice(self.Y.shape[0], 1, replace=False)

        mu_index = self.get_mu_index(self.X[index], self.mus_array1)
        if self.mus_label_array1[mu_index] == self.Y[index]:
            self.mus_array1[mu_index] = self.mus_array1[mu_index] + \
                                       self.eta * (self.X[index] - self.mus_array1[mu_index])
        else:
            self.mus_array1[mu_index] = self.mus_array1[mu_index] - \
                                       self.eta * (self.X[index] - self.mus_array1[mu_index])

        diff = 0
        for i in range(self.mus_array1.shape[0]):
            diff += np.linalg.norm(self.mus_array1[i] - old_mus_array[i])
        # if diff < self.e:
        #     print('迭代{}次退出'.format(iter))
        #     return

        mus = self.mus_array1
        plt.subplot(211)
        plt.scatter(mus[:, 0], mus[:, 1], marker='.', c='r')



        old_mus_array = copy.deepcopy(self.mus_array2)
        index = np.random.choice(self.Y2.shape[0], 1, replace=False)

        mu_index = self.get_mu_index(self.X2[index], self.mus_array2)
        if self.mus_label_array2[mu_index] == self.Y2[index]:
            self.mus_array2[mu_index] = self.mus_array2[mu_index] + \
                                       self.eta * (self.X2[index] - self.mus_array2[mu_index])
        else:
            self.mus_array2[mu_index] = self.mus_array2[mu_index] - \
                                       self.eta * (self.X2[index] - self.mus_array2[mu_index])

        diff = 0
        for i in range(self.mus_array2.shape[0]):
            diff += np.linalg.norm(self.mus_array2[i] - old_mus_array[i])
        # if diff < self.e:
        #     print('迭代{}次退出'.format(iter))
        #     return

        mus = self.mus_array2
        plt.subplot(212)
        plt.scatter(mus[:, 0], mus[:, 1], marker='.', c='r')

        # time.sleep(0.2)

def main():

    fig = plt.figure(1)

    plt.subplot(221)
    center = [[1, 1], [-1, -1], [1, -1]]
    cluster_std = 0.35
    X1, Y1 = make_blobs(n_samples=1000, centers=center,
                        n_features=2, cluster_std=cluster_std, random_state=1)

    plt.scatter(X1[:, 0], X1[:, 1], marker='o', c=Y1)

    plt.subplot(222)
    lvq1 = LVQ()
    lvq1.fit(X1, Y1)
    mus = lvq1.mus_array
    plt.scatter(X1[:, 0], X1[:, 1], marker='o', c=Y1)
    plt.scatter(mus[:, 0], mus[:, 1], marker='^', c='r')

    plt.subplot(223)
    X2, Y2 = make_moons(n_samples=1000, noise=0.1)
    plt.scatter(X2[:, 0], X2[:, 1], marker='o', c=Y2)

    plt.subplot(224)
    lvq2 = LVQ()
    lvq2.fit(X2, Y2)
    mus = lvq2.mus_array
    plt.scatter(X2[:, 0], X2[:, 1], marker='o', c=Y2)
    plt.scatter(mus[:, 0], mus[:, 1], marker='^', c='r')
    plt.show()



def generator_gif():

    fig = plt.figure(1)

    plt.subplot(211)
    center = [[1, 1], [-1, -1], [1, -1]]
    cluster_std = 0.35
    X1, Y1 = make_blobs(n_samples=1000, centers=center,
                        n_features=2, cluster_std=cluster_std, random_state=1)

    plt.scatter(X1[:, 0], X1[:, 1], marker='o', c=Y1)


    lvq1 = LVQ()
    # lvq1.fit(X1, Y1)
    # mus = lvq1.mus_array
    # plt.scatter(X1[:, 0], X1[:, 1], marker='o', c=Y1)
    # plt.scatter(mus[:, 0], mus[:, 1], marker='^', c='r')
    # plt.show()
    X2, Y2 = make_moons(n_samples=1000, noise=0.1)
    plt.subplot(212)
    plt.scatter(X2[:, 0], X2[:, 1], marker='o', c=Y2)

    lvq1.init_animate(X1, Y1, X2, Y2)
    # anim = FuncAnimation(fig, animate, init_func=init,
    #                      frames=200, interval=20, blit=True)
    anim = FuncAnimation(fig, lvq1.animate, interval=100)
    # anim.show()

    mus = lvq1.mus_array1
    plt.subplot(211)
    plt.scatter(mus[:, 0], mus[:, 1], marker='*', c='b')

    mus = lvq1.mus_array2
    plt.subplot(212)
    plt.scatter(mus[:, 0], mus[:, 1], marker='*', c='b')

    anim.save('/home/gswyhq/Downloads/lvq1.gif', writer='imagemagick')


# ————————————————
#
# 版权声明：本文为CSDN博主「无聊的六婆」的原创文章，遵循 CC 4.0 BY-SA 版权协议，转载请附上原文出处链接及本声明。
#
# 原文链接：https://blog.csdn.net/z962013489/article/details/82823932




if __name__ == '__main__':
    # main()
    generator_gif()
