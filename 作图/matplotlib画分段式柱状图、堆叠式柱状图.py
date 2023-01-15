#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# matplotlib实例一 分段式柱状图(堆叠式柱状图)

import matplotlib.pyplot as plt
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei'] #指定默认字体   
mpl.rcParams['axes.unicode_minus'] = False #解决保存图像是负号'-'显示为方块的问题

labels = ['A', 'B', 'C', 'D', 'E']
blue = [10, 20, 30, 40, 50]
yellow = [35, 15, 25, 20, 10]
blue_err = [1, 1, 1, 1, 1]
yellow_err = [1, 1, 1, 1, 1]

width = 0.5

fig = plt.figure(figsize=[10, 10], dpi=None)

ax = fig.add_subplot(1, 1, 1)  # 1行，1列，对1个子图


ax.bar(labels, blue, width, yerr=blue_err, label='blue',)
ax.bar(labels,yellow,width,yerr=yellow_err,bottom=blue,label='yellow')

ax.set_ylabel('Y')
ax.set_title('this is title')
ax.legend()  #显示图中左上角的标识区域
plt.ylim(5,55)  # 设置y坐标轴显示范围，极值范围
plt.show()


# 接口详解
# Axes.bar（self，x，height，width = 0.8，bottom = None，*，
# 		  align = 'center'，data = None，** kwargs）
#
# 参数	用法
# x	横坐标, 可以是一个值, 也可以是一个list
# height	柱状图高度, 可以是一个具体的值, 也可以是一个list
# width	柱体的宽度, 其他同上
# bottom	直译是底座, 这里表示柱状图下面是谁, 在实例中表现为, 黄色柱体的底下是蓝色的
# align	决定柱状图的位置, 两个参数可选, ‘center’(居中)，‘edge’(左边缘与x轴对齐)

def main():
    pass


if __name__ == '__main__':
    main()
