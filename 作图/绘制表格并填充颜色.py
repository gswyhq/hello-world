#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from matplotlib import cm
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei'] #指定默认字体
mpl.rcParams['axes.unicode_minus'] = False #解决保存图像是负号'-'显示为方块的问题
import numpy as np
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import pandas as pd

map_vir = cm.get_cmap('Pastel1')

data = [
    [100, 150, 50, 200],
    [200, 150, 150, 300],
    [50, 50, 50, 300],
]
col_labels = ['bbbbbbbbb1', '第二列', 'bbbbbbbbb3', 'bbbbbbbbb4']
row_labels = ['第一行', 'aaaaaaaa2', 'aaaaaaaa3']
low = 100
mid = 160
high = 200
cell_text = list()
cell_colors = list()
for row in data:
    row_text = list()
    row_color = list()
    for val in row:
        row_text.append('%.1f' % val)
        if val < low:
            color = 'g'
        elif low <= val < mid:
            color = 'y'
        elif mid <= val < high:
            color = 'b'
        else:  # val >= high
            color = 'r'
        row_color.append(color)
    cell_text.append(row_text)
    cell_colors.append(row_color)
print(cell_text)
# [['100.0', '150.0', '50.0', '200.0'], ['200.0', '150.0', '150.0', '300.0'], ['50.0', '50.0', '50.0', '300.0'], ['50.0', '90.0', '80.0', '70.0']]

print(cell_colors)
# [['y', 'y', 'g', 'r'], ['r', 'y', 'y', 'r'], ['g', 'g', 'g', 'r'], ['g', 'g', 'g', 'g']]

fig = plt.figure(figsize=(15, 4))  # 创建一个新的画布或引用一个已有的画布 figsize：设置画布的尺寸，默认 [6.4, 4.8]，单位英寸
ax = fig.subplots(nrows=1, ncols=1)
ax.set_axis_off()
ax.title.set_text('Table Title')
table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        rowLabels=row_labels,
        cellColours=cell_colors,
        cellLoc='center', rowLoc='center', loc='center'
    )

table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 2)
cellDict = table.get_celld()
for cell in cellDict:
    #print(cell)
    cellDict[cell].set_height(0.2)

plt.show()

# matplotlib之table参数解释详情
# 表格一般通常和其他图组合使用
# cellText:表格的数值，将源数据按照行进行分组，每组数据放在列表里存储，所有组数据再放在列表里储存。
# cellLoc:表格中的数据对齐位置，可以左对齐、居中和右对齐。
# colWidths:表格每列的宽度。
# colLabels:表格每列的列名称。
# colColours:表格每列的列名称所在单元格的颜色。
# rowLabels:表格每行的行名称。
# rowLoc:表格每行的行名称对齐位置，可以左对齐、居中和右对齐。（同理也有colLoc）
# loc:表格在画布中的位置。位置大致有以下：upper right、upper left、lower left、lower right、center left、center right、lower center、upper center、center、top right、top left、bottom left、bottom right、right、left、top、bottom

###########################################################################################################################
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams["font.sans-serif"] = ["SimHei"]  # 展示中文字体
mpl.rcParams["axes.unicode_minus"] = False  # 处理负刻度值
kinds = ["简易箱", "保温箱", "行李箱", "密封箱"]
colors = ["#e41a1c", "#377eb8", "#00ccff", "#984ea3"]
soldNums = [33, 98, 100, 50]

plt.pie(soldNums,
        labels=kinds,
        autopct="%3.1f%%",
        startangle=60,
        colors=colors)

# 饼图下添加表格
cellTexts = [soldNums]
rowLabels = ['价格']
plt.table(cellText=cellTexts,  # 简单理解为表示表格里的数据
          colWidths=[0.3] * 4,  # 每个小格子的宽度 * 个数，要对应相应个数
          colLabels=kinds,  # 每列的名称
          colColours=colors,  # 每列名称颜色
          rowLabels=rowLabels,  # 每行的名称（从列名称的下一行开始）
          rowLoc="center",  # 行名称的对齐方式
          loc="bottom"  # 表格所在位置
          )
plt.title("简单图形")
plt.figure(dpi=80)
plt.show()


def main():
    pass


if __name__ == '__main__':
    main()
