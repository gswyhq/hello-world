#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 2、散点图不同 颜色上色/散点大小 的方法
# ① 数据中有一列专门用于设置颜色 / 点大小

from bokeh.palettes import brewer
import numpy as np
import pandas as pd

# https://www.cnblogs.com/shengyang17/p/9736757.html

rng = np.random.RandomState(1)
df = pd.DataFrame(rng.randn(100,2)*100,columns = ['A','B'])
# 创建数据，有2列随机值

df['size'] = rng.randint(10,30,100)   # 设置点大小字段

# colormap1 = {1: 'red', 2: 'green', 3: 'blue'}
# df['color1'] = [colormap1[x] for x in rng.randint(1,4,100)]           # 调色盘1；
df['color1'] = np.random.choice(['red', 'green', 'blue'], 100) #跟上面两行是一样的；  这两种都是在本身的数据中增加size和color1标签，再去绘制图标；

print(df.head())

p = figure(plot_width=600, plot_height=400)
p.circle(df['A'], df['B'],       # 设置散点图x，y值
         line_color = 'white',   # 设置点边线为白色
         fill_color = df['color1'],fill_alpha = 0.5,   # 设置内部填充颜色，这里用到了颜色字段
         size = df['size']       # 设置点大小，这里用到了点大小字段,按照size的随机数去设置点的大小
        )

show(p)

def main():
    pass


if __name__ == '__main__':
    main()