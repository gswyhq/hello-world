#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

# 若将散点大小的数据换为第三个变量的数值，则可以作出反映三个变量关系的气泡图。下面的代码和图形做出了一个气泡图。下图反映了产量与温度、降雨量的关系：温度数值在横坐标轴，降雨量数值在纵坐标轴，降雨量的大小用气泡的大小表示。

# 这两行代码解决 plt 中文显示的问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 输入产量与温度数据
production = [1125, 1725, 2250, 2875, 2900, 3750, 4125]
tem = ['6℃', '8℃', '10℃', '13℃', '14℃', '16℃', '21℃'] # [6, 8, 10, 13, 14, 16, 21]
rain = ['25ml', '40ml', '58ml', '68ml', '110ml', '98ml', '120ml'] # [25, 40, 58, 68, 110, 98, 120]

colors = np.random.rand(len(tem))  # 颜色数组
size = production
plt.scatter(tem, rain, s=size, c=colors, alpha=0.6)  # 画散点图, alpha=0.6 表示不透明度为 0.6
# plt.ylim([0, 150])  # 纵坐标轴范围
# plt.xlim([0, 30])   # 横坐标轴范围
plt.xlabel('温度')  # 横坐标轴标题
plt.ylabel('降雨量')  # 纵坐标轴标题
plt.show()

def main():
    pass


if __name__ == '__main__':
    main()