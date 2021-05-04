#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as NP
from matplotlib import pyplot as PLT
from matplotlib import cm as CM
from matplotlib import axes
A = NP.array([
 [6.55,6.76,7.32,5.6,5.94,],
 [0.01,0.01,0.04,0.02,0.11,],
 [6.45,6.29,4.34,4.57,7.15,],
 [8.73,10.67,6.9,8.25,8.53,],
 [0.03,0.01,0.05,0.01,0.07,],
 [1.36,1.41,0.8,0.98,1.36,],
 [0,0,0,0,0.01,],
 [2.09,2.93,1.94,1.96,3.56,],
 [3.61,3.37,2.62,6.99,4.6,],
 [6.08,7.04,4.72,4.78,7.2,],
 [1.67,0.92,0.62,3.87,0.75,],
 [0.01,0,0,0.11,0.18,],
 [0.03,0,0.13,0.03,0.24,],
 [1.66,1.8,1.81,1.81,2.12,],
 [3.37,3.48,4.39,3.02,3.2,],
 [4.77,4.91,8.62,5.35,4.68,],
 [7.58,7.9,3.02,7.57,8.15,],
 [7.59,7.79,9.92,8.17,7.61,],
 [3.59,3.46,2.54,2.99,2.68,],
 [2.51,3.82,4.57,2.56,3.19,],
 [1.74,2.38,5.4,2.05,2.24,],
 [4.71,3.05,5.12,2.73,4.18,],
 [0.85,0.93,2.47,0.83,1.12,],
 [11.62,12.01,10.43,12.49,12.42,],
 [13.4,9.06,12.24,13.26,8.71,],
 ])

# 设定一个图像，背景为白色。
fig = PLT.figure(facecolor='w')
#注意位置坐标，数字表示的是坐标的比例
ax1 = fig.add_subplot(2,1,1,position=[0.1,0.15,0.9,0.8])
#注意标记旋转的角度
ax1.set_xticklabels(labels=['','A','B','C','D','E'], rotation=-45 )

# select the color map
#可以有多种选择，这里我最终选择的是spectral，那个1000是热度标尺被分隔成多少块，数字越多，颜色区分越细致。
#cmap = CM.get_cmap('RdYlBu_r', 1000)
cmap = CM.get_cmap('rainbow', 1000)
# cmap = CM.get_cmap('Spectral', 1000)

# map the colors/shades to your data
#那个vmin和vmax是数据矩阵中的最大和最小值。这个范围要与数据的范围相协调。
#那个aspect参数，对确定图形在整个图中的位置和大小有关系。上面的add_subplot中的position参数的数值要想有作用，这里的这个参数一定要选auto。
map = ax1.imshow(A, interpolation="nearest", cmap=cmap,aspect='auto', vmin=0,vmax=15)
#shrink是标尺缩小的比例
cb = PLT.colorbar(mappable=map, cax=None, ax=None,shrink=0.5)
cb.set_label('(%)')

# plot it
PLT.show()

def main():
    pass


if __name__ == '__main__':
    main()