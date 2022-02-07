#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文(windows)
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

df_0912 = [random.randint(20, max(25, i%80)) for i in range(480)]
df_0915 = [random.randint(20, 60)+i%10 for i in range(480)]
df_0916 = [random.randint(30, 40) for i in range(480)]

# 生成时间序列：X轴刻度数据
table = pd.DataFrame([i for i in range(480)], columns=['value'],
                     index=pd.date_range('00:00:00', '23:57:00', freq='180s'))

# 二、绘制图形
# 图片大小设置
fig = plt.figure(figsize=(15, 9), dpi=100)
ax = fig.add_subplot(111)

# X轴时间刻度格式 & 刻度显示
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.xticks(pd.date_range(table.index[0], table.index[-1], freq='H'), rotation=45)

# 绘图
ax.plot(table.index, df_0912, color='r', label='9月12日')
ax.plot(table.index, df_0915, color='y', label='9月15日')
ax.plot(table.index, df_0916, color='g', label='9月16日')

# 给某个点添加注释， 单点注释
y0 = max(df_0912)
x0 = table.index[df_0912.index(y0)]
plt.annotate(r"最大值点", xy=(x0, y0), xycoords='data', xytext=(+30, -30),
             textcoords='offset points', arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))

# 辅助线
sup_line = [35 for i in range(480)]
ax.plot(table.index, sup_line, color='black', linestyle='--', linewidth=1, label='辅助线')

plt.xlabel('time_point', fontsize=14)  # X轴标签
plt.ylabel("Speed", fontsize=16)  # Y轴标签
ax.legend()  # 图例
plt.title("时序图", fontsize=25, color='black', pad=20)
plt.gcf().autofmt_xdate()

# 隐藏-上&右边线
# ax.spines['right'].set_color('none')
# ax.spines['top'].set_color('none')

# plt.savefig('speed.png')
plt.show()

def main():
    pass


if __name__ == '__main__':
    main()