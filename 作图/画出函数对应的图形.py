#!/usr/bin/python3
# coding: utf-8

import math
import pylab as pl
import matplotlib.pyplot as plt

def show_plot(x_list, y_list):
    # 设置xy,边界
    plt.xlim(xmax=max(x_list)+0.5,xmin=min(x_list)-0.5)
    # plt.ylim(ymax=7,ymin=0)

    # 加点标注
    # plt.annotate("(3,6)", xy = (3, 6), xytext = (4, 5), arrowprops = dict(facecolor = 'black', shrink = 0.1))

    # x, y轴命名
    plt.xlabel("x")
    plt.ylabel("y")
    plt.plot(x_list,y_list,'ro')
    plt.show()


def sigmoid_f(x, a=None):
    if a is None:
        return 1 / (1 + math.exp(-x))
    else:
        return 1/ (1 + a**(-x))

def main():
    x_list = [i for i in pl.frange(-20, 20, 0.1)]
    # y_list = [sigmoid_f(x) for x in x_list]
    # show_plot(x_list, y_list)


    y_list = [sigmoid_f(x, 1.1894132348229451) for x in x_list]
    show_plot(x_list, y_list)

if __name__ == '__main__':
    main()