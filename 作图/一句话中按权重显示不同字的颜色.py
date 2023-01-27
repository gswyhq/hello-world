#!/usr/bin/env python3
# -*- coding: utf-8 -*-

result_list = [[[('这', 1.0), ('家', 0.5968), ('店', 0.5769), ('真', 0.7146), ('黑', 0.6856), ('心', 0.4891)], [('这', 1.0), ('家', 0.4878), ('店', 0.4308), ('真', 0.5372), ('黑', 0.6127), ('心', 0.3741)]], [[('图', 0.3615), ('太', 0.51), ('乱', 0.5184), ('了', 0.7016), ('有', 0.5052), ('点', 0.7518), ('看', 0.5037), ('不', 1.0), ('懂', 0.5813), ('重', 0.5008), ('点', 0.7518)], [('图', 0.2857), ('太', 0.4945), ('乱', 0.4426), ('了', 0.9089), ('有', 0.4122), ('点', 0.7033), ('看', 0.4722), ('不', 1.0), ('懂', 0.7469), ('重', 0.8275), ('点', 0.7033)]], [[('讲', 0.3337), ('故', 0.4751), ('事', 0.4856), ('的', 0.4192), ('时', 0.3636), ('候', 0.4403), ('很', 1.0), ('难', 0.9525), ('让', 0.6668), ('孩', 0.4725), ('子', 0.1931), ('集', 0.2623), ('中', 0.3146)], [('讲', 0.2119), ('故', 0.4284), ('事', 0.4665), ('的', 0.3327), ('时', 0.2989), ('候', 0.4666), ('很', 1.0), ('难', 0.8644), ('让', 0.6868), ('孩', 0.5733), ('子', 0.1824), ('集', 0.1948), ('中', 0.2494)]], [[('这', 1.0), ('是', 0.899), ('一', 0.5212), ('本', 0.7363), ('很', 0.9652), ('好', 0.8309), ('看', 0.9523), ('的', 0.6179), ('书', 0.4557)], [('这', 0.7845), ('是', 0.787), ('一', 0.3255), ('本', 0.5876), ('很', 0.8018), ('好', 0.6167), ('看', 1.0), ('的', 0.3475), ('书', 0.2783)]], [[('这', 0.8185), ('是', 0.5494), ('一', 0.3393), ('本', 0.5586), ('很', 0.976), ('糟', 1.0), ('糕', 0.4005), ('的', 0.5), ('书', 0.43)], [('这', 0.7943), ('是', 0.4889), ('一', 0.2518), ('本', 0.4996), ('很', 1.0), ('糟', 0.7303), ('糕', 0.2435), ('的', 0.3136), ('书', 0.35)]]]

import matplotlib.pyplot as plt
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei'] #指定默认字体
mpl.rcParams['axes.unicode_minus'] = False #解决保存图像是负号'-'显示为方块的问题
import numpy as np
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import pandas as pd
ax = plt.subplot(111)  # 注意:一般都在ax中设置,不再plot中设置

# 颜色由红到白
# color_list = ['#ff0000', '#ff1111', '#ff2222', '#ff3333', '#ff4444', '#ff5555', '#ff6666', '#ff7777', '#ff8888', '#ff9999', '#ffaaaa', '#ffbbbb', '#ffcccc', '#ffdddd', '#ffeeee', '#ffffff']

for row, result in enumerate(result_list):
    x2 = np.linspace(row, row+1, 1)
    # ax.fill_between(x2, row, row+1, facecolor=color_list[row])
    max_pred = max([t[-1] for t in result[0]])
    for col, ret in enumerate(result[0]):
        word = ret[0]
        pred = ret[1]
        # 更多颜色设置参考：
        # # https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
        # print("最浅颜色（rgba四通道）：{}".format(plt.get_cmap("Reds").get_under()))
        # print("最深颜色（rgba四通道）：{}".format(plt.get_cmap("Reds").get_over()))
        # facecolor='green', '#ff0000', (1, 0, 0)
        facecolor = (1, 1-pred/max_pred, 1-pred/max_pred)
        ax.text(col, row, word, bbox=dict(facecolor=facecolor, alpha=0.5))

plt.xlim(0, 15)
plt.ylim(0, 5)
ax.xaxis.set_major_locator(MultipleLocator(1))  # 设置y主坐标间隔 1
ax.yaxis.set_major_locator(MultipleLocator(1))  # 设置y主坐标间隔 1
ax.xaxis.grid(True, which='major')  # major,color='black'
ax.yaxis.grid(True, which='major')  # major,color='black'
plt.show()

def main():
    pass


if __name__ == '__main__':
    main()
