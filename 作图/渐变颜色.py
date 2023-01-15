#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors

class SqueezedNorm(matplotlib.colors.Normalize):
    def __init__(self, vmin=None, vmax=None, mid=0, s1=2, s2=2, clip=False):
        self.vmin = vmin # minimum value
        self.mid  = mid  # middle value
        self.vmax = vmax # maximum value
        self.s1=s1; self.s2=s2
        f = lambda x, zero,vmax,s: np.abs((x-zero)/(vmax-zero))**(1./s)*0.5
        self.g = lambda x, zero,vmin,vmax, s1,s2: f(x,zero,vmax,s1)*(x>=zero) - \
                                             f(x,zero,vmin,s2)*(x<zero)+0.5
        matplotlib.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        r = self.g(value, self.mid,self.vmin,self.vmax, self.s1,self.s2)
        return np.ma.masked_array(r)


fig, (ax, ax2, ax3) = plt.subplots(nrows=3,
                                   gridspec_kw={"height_ratios":[3,2,1], "hspace":0.25})

x = np.linspace(-13, 4, 110)

norm=SqueezedNorm(vmin=-13, vmax=4, mid=0, s1=1.7, s2=4)

line, = ax.plot(x, norm(x))
ax.margins(0)
ax.set_ylim(0,1)

im = ax2.imshow(np.atleast_2d(x).T, cmap="Spectral_r", norm=norm, aspect="auto")
cbar = fig.colorbar(im ,cax=ax3,ax=ax2, orientation="horizontal")

##################################################################################
# 设置 mid = np.mean(vmin, vmax), s1=1, s2=1 将恢复原始缩放。
fig, (ax, ax2, ax3) = plt.subplots(nrows=3,
                                   gridspec_kw={"height_ratios":[3,2,1], "hspace":0.25})

x = np.linspace(-13, 4, 110)

norm=SqueezedNorm(vmin=-13, vmax=4, mid = np.mean([-13, 4]), s1=1, s2=1)

line, = ax.plot(x, norm(x))
ax.margins(0)
ax.set_ylim(0,1)

im = ax2.imshow(np.atleast_2d(x).T, cmap="Spectral_r", norm=norm, aspect="auto")
cbar = fig.colorbar(im ,cax=ax3,ax=ax2, orientation="horizontal")

###########################################################################################################

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Create custom colormaps
cdict = {'red': ((0.0, 1.0, 1.0),   # Full red at the first stop
                 (0.5, 0.0, 0.0),   # No red at second stop
                 (1.0, 1.0, 1.0)),  # Full red at final stop
        #
        'green': ((0.0, 0.0, 0.0),  # No green at all stop
                 (0.5, 0.0, 0.0),   #
                 (1.0, 0.0, 0.0)),  #
        #
        'blue': ((0.0, 0.0, 0.0),   # No blue at first stop
                 (0.5, 1.0, 1.0),   # Full blue at second stop
                 (1.0, 0.0, 0.0))}  # No blue at final stop

cmap = LinearSegmentedColormap('Rd_Bl_Rd', cdict, 256)
im = np.outer(np.ones(10), np.linspace(0, 255, 256))
fig = plt.figure(figsize=(9, 2))
ax = fig.add_subplot('111')
ax.set_xticks(np.linspace(0, 255, 3))
ax.set_xticklabels([0, 0.5, 1])
ax.set_yticks([])
ax.set_yticklabels([])
ax.imshow(im, interpolation='nearest', cmap=cmap)
###########################################################################################################

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from collections import OrderedDict

cmaps = OrderedDict()
print("全部配色方案：{}".format(plt.colormaps()))

cmaps['Perceptually Uniform Sequential'] = [
            'viridis', 'plasma', 'inferno', 'magma', 'cividis']

cmaps['Sequential'] = [
            'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']
cmaps['Sequential (2)'] = [
            'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
            'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
            'hot', 'afmhot', 'gist_heat', 'copper']

cmaps['Diverging'] = [
            'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
            'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic']
cmaps['Cyclic'] = ['twilight', 'twilight_shifted', 'hsv']

cmaps['Qualitative'] = ['Pastel1', 'Pastel2', 'Paired', 'Accent',
                        'Dark2', 'Set1', 'Set2', 'Set3',
                        'tab10', 'tab20', 'tab20b', 'tab20c']
cmaps['Miscellaneous'] = [
            'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
            'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg',
            'gist_rainbow', 'rainbow', 'jet', 'nipy_spectral', 'gist_ncar']

nrows = max(len(cmap_list) for cmap_category, cmap_list in cmaps.items())
gradient = np.linspace(0, 1, 256)
gradient = np.vstack((gradient, gradient)) # np.vstack() 按垂直方向(行顺序)堆叠数组构成一个新的数组,堆叠的数组需要具有相同的维度


def plot_color_gradients(cmap_category, cmap_list, nrows):
    fig, axes = plt.subplots(nrows=nrows)
    fig.subplots_adjust(top=0.95, bottom=0.01, left=0.2, right=0.99)
    axes[0].set_title(cmap_category + ' colormaps', fontsize=14)

    for ax, name in zip(axes, cmap_list):
        ax.imshow(gradient, aspect='auto', cmap=plt.get_cmap(name))
        pos = list(ax.get_position().bounds)
        x_text = pos[0] - 0.01
        y_text = pos[1] + pos[3]/2.
        fig.text(x_text, y_text, name, va='center', ha='right', fontsize=10)

    # Turn off *all* ticks & spines, not just the ones with colormaps.
    for ax in axes:
        ax.set_axis_off()


for cmap_category, cmap_list in cmaps.items():
    plot_color_gradients(cmap_category, cmap_list, nrows)

plt.show()

#######################################################################################################################
from matplotlib.colors import LinearSegmentedColormap
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei'] #指定默认字体
mpl.rcParams['axes.unicode_minus'] = False #解决保存图像是负号'-'显示为方块的问题
import numpy as np
import matplotlib.pyplot as plt
tem = np.random.random(size=(3,5))
name = 'Reds'  # 选中一个渐变色
cmap =  LinearSegmentedColormap('cmap', plt.get_cmap(name)._segmentdata, 256)  # 这里分级显示，分为256级；
plt.imshow(tem,cmap = cmap)  # plt.imshow()作用就是展示一副热度图，将数组表示为一幅图
plt.ylim(ymax=2.5,ymin=-0.5)  # 设置Y轴边界范围
for col in range(5):
    plt.text(col, 0, 'center_{}'.format(col), ha='center')  # 热度图中填充文本
plt.colorbar(label='颜色渐变范围')

plt.xticks(range(5), list('abcde'))  # 设置X坐标轴刻度名称
plt.yticks(range(3), ['1行', '2行', '3行'])  # 设置y坐标轴刻度名称
plt.show()

# 更多参考：
# https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html

# 可以将 cmap 输出到HTML文件中，查看；
with open(r"123.html", 'w', encoding='utf-8')as f:
    f.write(cmap._repr_html_())
print("最浅颜色（rgba四通道）：{}".format(cmap.get_under()))
print("最深颜色（rgba四通道）：{}".format(cmap.get_over()))

def main():
    pass


if __name__ == '__main__':
    main()
