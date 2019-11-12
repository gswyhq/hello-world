#!/usr/bin/python3
# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm
from matplotlib import rc

myfont = fm.FontProperties(fname='/usr/share/fonts/deepin-font-install/SimSun/SimSun.ttf')

# 中文字体乱码，matplotlib各项设置都是正确的，但还是显示中文乱码，可能是因为缓存所致
# 删除当前用户matplotlib 的缓冲文件(这步很重要)
# rm ~/.cache/matplotlib/*


print('matplotlib缓存文件路径：{}'.format(fm.get_cachedir()))

font_name = myfont.get_name()
rc('font', family = font_name)

# 构建数据
x_data = ['2011', '2012', '2013', '2014', '2015', '2016', '2017']
y_data = [58000, 60200, 63000, 71000, 84000, 90500, 107000]
y_data2 = [52000, 54200, 51500,58300, 56800, 59500, 62700]
bar_width=0.3
# 将X轴数据改为使用range(len(x_data), 就是0、1、2...

if '柱状图并列':
    plt.bar(x=range(len(x_data)), height=y_data, label='C语言基础',
        color='steelblue', alpha=0.8, width=bar_width)

    # 将X轴数据改为使用np.arange(len(x_data))+bar_width,
    # 就是bar_width、1+bar_width、2+bar_width...这样就和第一个柱状图并列了

    plt.bar(x=np.arange(len(x_data))+bar_width, height=y_data2,
        label='Java基础', color='indianred', alpha=0.8, width=bar_width)

elif '累计柱状图':
    plt.bar(x=x_data, height=y_data, label='C语言基础', color='steelblue', alpha=0.8)
    plt.bar(x=x_data, height=y_data2, label='Java基础', color='indianred', alpha=0.8)

# 在柱状图上显示具体数值, ha参数控制水平对齐方式, va控制垂直对齐方式
for x, y in enumerate(y_data):
    plt.text(x, y + 100, '%s' % y, ha='center', va='bottom')
for x, y in enumerate(y_data2):
    plt.text(x+bar_width, y + 100, '%s' % y, ha='center', va='top')
# 设置标题
plt.title("C与Java对比", fontproperties=myfont)
# 为两条坐标轴设置名称
plt.xlabel("年份", fontproperties=myfont)
plt.ylabel("销量", fontproperties=myfont)
# 显示图例
plt.legend()
plt.show()


def main():
    pass


if __name__ == '__main__':
    main()