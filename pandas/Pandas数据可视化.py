#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt
from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']
# mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei'] # 指定默认字体：解决plot不能显示中文问题
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

df = pd.DataFrame(np.random.rand(10, 4), columns=['A', 'B', 'C', 'D'])
# 01、柱状图 - 纵向

df.plot.bar(title='柱状图-纵向')

# stacked = True，画堆叠柱状图

df.plot.bar(stacked=True, title='堆叠柱状图')

# 02、柱状图 - 横向

df.plot.barh(title='柱状图-横向')

# 同样，stacked = True，画堆叠柱状图

df.plot.barh(stacked=True, title='柱状图-横向-堆叠柱状图')

# 03、面积图

df.plot.area(stacked=False, alpha=0.9, title='面积图')

df.plot.area(stacked=True, alpha=0.9, title='面积图-纵向')

# 04、密度图 - kde

df.plot.kde(title='密度图-kde')

# 05、密度图 - density

df.plot.density(title='密度图-density')

# 06、直方图

# 换个数据集

df = pd.DataFrame({'A': np.random.randn(1000) + 1,
                   'B': np.random.randn(1000),
                   'C': np.random.randn(1000) - 1},
                  columns=['A', 'B', 'C'])
df.plot.hist(bins=200, title='直方图')

df.plot.hist(stacked=True, bins=20, title='直方图2')

df = pd.DataFrame(np.random.rand(1000, 4), columns=['A', 'B', 'C', 'D'])
df.diff().hist(color='k', alpha=0.7, bins=50)

# 07、箱盒图

df = pd.DataFrame(np.random.rand(100, 4), columns=['A', 'B', 'C', 'D'])
df.plot.box(title='箱盒图')

# vert = False也可以换成横向

df.plot.box(title='横向箱盒图', vert=False)

# 8、散点图

df.plot.scatter(title='散点图', x='A', y='B')

# 9、蜂巢图

df = pd.DataFrame(np.random.randn(1000, 2), columns=['a', 'b'])
df['b'] = df['b'] + np.arange(1000)
df.plot.hexbin(title='蜂巢图', x='a', y='b', gridsize=25)

# 07、饼图

series = pd.Series(3 * np.random.rand(4), index=['a', 'b', 'c', 'd'], name='series')
series.plot.pie(figsize=(6, 6), title='饼状图')

series.plot.pie(labels=['AA', 'BB', 'CC', 'DD'], colors=['r', 'g', 'b', 'c'], autopct='%.2f', fontsize=20,
                figsize=(6, 6), title='饼状图2')


# 8、矩阵散点图

from pandas.plotting import scatter_matrix
df = pd.DataFrame(np.random.randn(1000, 4), columns=['a', 'b', 'c', 'd'])
scatter_matrix(df, alpha=0.2, figsize=(6, 6), diagonal='kde')

# 9、安德鲁斯曲线

# https://github.com/mwaskom/seaborn-data/blob/master/iris.csv
# 获取数据集

data = pd.read_csv(r'D:\Users\gswyhq\data\iris.csv')
plt.figure('安德鲁斯曲线')
plt.title('安德鲁斯曲线')
pd.plotting.andrews_curves(data, 'species')
plt.show()
time.sleep(10)
plt.close()

plt.figure('安德鲁斯曲线')
plt.title('安德鲁斯曲线2')
pd.plotting.andrews_curves(data, 'species', colormap='winter')
plt.show()
time.sleep(10)
plt.close()

# 10、平行坐标图

# 该图也是使用自己加载的iris数据集

from pandas.plotting import parallel_coordinates
plt.figure('平行坐标图')
plt.title('平行坐标图')
parallel_coordinates(data, 'species', colormap='gist_rainbow')
plt.show()

# 11、Lag  Plot

from pandas.plotting import lag_plot

df = pd.Series(0.1 * np.random.rand(1000) +
               0.9 * np.sin(np.linspace(-99 * np.pi, 99 * np.pi, num=1000)))
lag_plot(df)

# 12、默认函数plot
# 直接画图，默认为折线图

df = pd.DataFrame(np.random.rand(12, 4), columns=['A', 'B', 'C', 'D'])
df.plot()

df.plot(subplots=True, layout=(2, 2), figsize=(15, 8))

df = pd.DataFrame(np.random.rand(1000, 4), columns=['A', 'B', 'C', 'D'])
df.plot()

df.plot(subplots=True, layout=(2, 2), figsize=(15, 8))

# 13、bootstrap_plot

s = pd.Series(np.random.uniform(size=100))
pd.plotting.bootstrap_plot(s)

# 来源：
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.plot.html
# https://blog.csdn.net/weixin_41666747/article/details/110507852

def main():
    pass


if __name__ == '__main__':
    main()