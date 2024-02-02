#!/usr/bin/python3
# coding: utf-8

import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams["font.sans-serif"] = ["SimHei"]  # 展示中文字体
mpl.rcParams["axes.unicode_minus"] = False  # 处理负刻度值

plt.title("I'm a scatter diagram.")

# 设置xy,边界
plt.xlim(xmax=7,xmin=0)
plt.ylim(ymax=7,ymin=0)

# 加点标注
plt.annotate("(3,6)", xy = (3, 6), xytext = (4, 5), arrowprops = dict(facecolor = 'black', shrink = 0.1))

# x, y轴命名
plt.xlabel("x")
plt.ylabel("y")
plt.plot([1,2,3],[4,5,6],'ro')
plt.show()

def test2():
    plt.subplot(221)
    plt.xlim(xmax=7, xmin=0)
    plt.ylim(ymax=7, ymin=0)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.plot([1, 2, 3], [4, 5, 6], 'ro')

    plt.subplot(222)
    plt.xlim(xmax=7, xmin=0)
    plt.ylim(ymax=7, ymin=0)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.plot([1, 2, 3], [4, 5, 6], 'ro')

    plt.subplot(223)
    plt.xlim(xmax=7, xmin=0)
    plt.ylim(ymax=7, ymin=0)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.plot([1, 2, 3], [4, 5, 6], 'ro')

    plt.show()

def main():
    pass


if __name__ == '__main__':
    main()
