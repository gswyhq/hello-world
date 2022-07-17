
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as ticker
import matplotlib as mpl

variables = ['A','B','C','X']
labels = ['ID_0','ID_1','ID_2','ID_3']

mpl.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei'] # 设置中文字体为黑体
mpl.rcParams['axes.unicode_minus'] = False


d = [[1, 2, 3, 4], [0.2, 8, 8, 4], [-2, 0, -3, 2], [6, 6, 4, 1]]
df = pd.DataFrame(d, columns=variables, index=labels)  #其中d为4*4的矩阵
fig = plt.figure(figsize=(15,15))    #设置图片大小
ax = fig.add_subplot(111)
cax = ax.matshow(df, interpolation='nearest', cmap='hot_r')
fig.colorbar(cax)

tick_spacing = 1
ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

ax.set_xticklabels([''] + list(df.columns))
ax.set_yticklabels([''] + list(df.index))

plt.show()

