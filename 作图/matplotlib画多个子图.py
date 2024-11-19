import glob
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

import matplotlib.ticker as mticker

from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei'] #指定默认字体   
mpl.rcParams['axes.unicode_minus'] = False #解决保存图像是负号'-'显示为方块的问题

def show_fig(ax, name, fun_type, kwargs, x_min, x_max, fontsize=1):
    x = np.arange(x_min, x_max, (x_max-x_min)/100)
    a = kwargs.get('a', 0)
    b = kwargs.get('b', 0)
    k = kwargs.get('k', 0)
    if fun_type == '设置分类权重':
        x = []
        height = []
        labels = []
        for index, (k, v ) in enumerate(kwargs.items(), 1):
            x.append(index)
            height.append(v)
            labels.append(k)
        # print(x, height, labels)
        ax.bar(x, height, align='center', alpha=0.7, width=0.2)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
    elif fun_type == '指数函数':
        y = [(a**(i/b)-1)/(a-1) for i in x]
        plt.plot(x, y, label="y=({}^(x/{})-1)/{}".format(a, b, a-1))
    elif fun_type == 'sigmoid函数':
        y = [1 / (1 + math.exp(-k * (t - a))) for t in x]
        plt.plot(x, y, label="1/(1+exp(-{}*(x-{})))".format(k, a))
    elif fun_type == '对数函数':
        y = [math.log(t -b + 1, a) for t in x]
        plt.plot(x, y, label="y=log {}(x+1)".format(a))
    else:
        raise ValueError("不支持的函数类型：{}".format(fun_type))

    # ax.set_xlabel(name, fontsize=fontsize, color='red')
    ax.set_ylabel("相对价值系数", fontsize=fontsize)
    ax.set_title(name, fontsize=fontsize, color='#fc7100', y=0.8)
    # plt.ylim(0, 1)
    ax.legend()
    return ax

def generator_map_function():
    setup_datas = [['字段数', '对数函数', {'a': 128, 'b': 0}, 1, 130],
                 ['记录数', '对数函数', {'a': 1e9, 'b': 0}, 0, 1e9],
                 ['饱和度', 'sigmoid函数', {'a': 0.5,'k': 10}, 0, 1],
                 ['正确性', 'sigmoid函数', {'a': 0.5,'k': 10}, 0, 1],
                 ['一致度', 'sigmoid函数', {'a': 0.5,'k': 10}, 0, 1],
                 ['距今天数', 'sigmoid函数', {'a': 182,'k': -0.027}, 0, 365],
                 ['更新频率', '设置分类权重', {'实时': 1.0, '日频': 0.95, '周频': 0.8, '月频': 0.7, '季频': 0.6, '年频': 0.4, '不更新': 0.3}, 0, 1],
                 ['关联表数', '指数函数', {'b': 8, 'a': 15}, 1, 9],
                 ['查看频率', '对数函数', {'a': 200, 'b': 0}, 1, 210],
                 ['交易频率', '对数函数', {'a': 100, 'b': 0}, 1, 110],
                 ['调用频率', '对数函数', {'a': 1000, 'b': 0}, 1, 1010],
                 ['查看量', '对数函数', {'a': 800, 'b': 0}, 1, 810],
                 ['交易量', '对数函数', {'a': 400, 'b': 0}, 1, 410],
                 ['调用量', '对数函数', {'a': 4000, 'b': 0}, 1, 4010],
                 ['稀缺度', 'sigmoid函数', {'a': 0.5, 'k': 10}, 0, 1],
                 ['安全分级', '设置分类权重', {'1级': 0.2, '2级': 0.4, '3级': 0.6, '4级': 0.8, '5级': 1.0}, 0, 1],
                 ['业务类别', '设置分类权重', {'分析工具': 0.6, '商家位置': 0.95, '专业排名': 0.75, '城市商圈': 0.85, '城市小区': 0.9, '城市房价': 0.85, '高校排名': 0.7}, 0, 1],
                 ['收入金额', '对数函数', {'a': 100, 'b': 0}, 0, 110]]


    fig = plt.figure()

    for index, (name, fun_type, kwargs, x_min, x_max) in enumerate(setup_datas, 1):
        print(name)
        ax = fig.add_subplot(6, 3, index) # 6行3列个子图，第一个子图index= 1；

        show_fig(ax, name, fun_type, kwargs, x_min, x_max, fontsize=9)
    # plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=2)
    # plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace=0)
    fig.suptitle("指标值的映射曲线", fontsize=12, x=0.5, y=0.92)
    fig.show()

# 设置子图x轴标签：
# 方法1：使用子图对象的轴方法（例如ax.set_xticks和ax.set_xticklabels）

# ax1.set_xticks([0,250,500,750,1000]) # 设置刻度
# ax1.set_xticklabels(['one','two','three','four','five'],rotation = 30,fontsize = 'small') # 设置刻度标签

# 方法2：使用plt.sca设置pyplot状态机的当前轴（即plt接口）。

import matplotlib.pyplot as plt

fig, axes = plt.subplots(nrows=3, ncols=4)

# Set the ticks and ticklabels for all axes
plt.setp(axes, xticks=[0.1, 0.5, 0.9], xticklabels=['a', 'b', 'c'],
        yticks=[1, 2, 3])

# Use the pyplot interface to change just one subplot...
plt.sca(axes[1, 1])
plt.xticks(range(3), ['A', 'Big', 'Cat'], color='red')

fig.tight_layout()
plt.show()

####################################################################################################################################
# 多子图双坐标系
ds  = {"train": {"label_names": ['11~20岁', '21~30岁', '31~40岁', '41~50岁', '51~60岁', '61~70岁', '71~80岁', '81~90岁'],
                 "label_values": [9, 1055, 6315, 5224, 2562, 610, 83, 7],
                 "label_rate": [0.00025, 0.03925, 0.2141875, 0.166125, 0.06975, 0.009875, 0.0, 0.0]
                }, 
      "dev":{ "label_names": ['11~20岁', '21~30岁', '31~40岁', '41~50岁', '51~60岁', '61~70岁', '71~80岁', '81~90岁', '90岁以上'],
              "label_values": [5, 276, 1550, 1292, 642, 166, 27, 3, 1],
              "label_rate": [0.00075, 0.042, 0.20625, 0.1655, 0.06975, 0.011, 0.00025, 0.0, 0.0]
            }
      }

def df2ax(ds, ax1):
    label_names = ds['label_names']
    label_values = ds['label_values']
    label_rate = ds['label_rate']
    ax1.bar(label_names, label_values, color="b", alpha=0.5)  # 柱形图
    ax2 = ax1.twinx() # 双坐标系
    ax2.plot(label_names, label_rate, color="y")  # 折线图
    fig.legend(['频数', '占比'], loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes) # 定义图例

fig, axs = plt.subplots(figsize=(16, 8), nrows=1, ncols=2) # 定义一行两列，共两个子图
plt.xticks(rotation=45)
df2ax(ds['train'], axs[0])
df2ax(ds['dev'], axs[1])
plt.show()

####################################################################################################################################
    
def main():
    generator_map_function()



if __name__ == '__main__':
    main()

