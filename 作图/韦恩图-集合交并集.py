#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#  在两个集合的韦恩图中，首先，可以有两个（或更多）重叠的圆圈，分别代表不同大小的集合，但是圆圈的大小相同。
#  实际上，圆圈应与集合的大小成比例，重叠区域也应与数据的重叠成比例。这样，在注意到数字之前，您可以立即看到重叠的数字。

# 参考： https://blog.csdn.net/luohenyj/article/details/103091081

# 导入必要的包
# 如果本地没有包的话，使用pip安装即可
# pip install matplotlib_venn
# matplotlib_venn, 仅仅支持2组和3组数据的韦恩图；
# pyvenn支持2到6组数据；https://github.com/tctianchi/pyvenn

import matplotlib.pyplot as plt
from matplotlib_venn import venn3, venn2
import matplotlib.patheffects as path_effects

from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei'] #指定默认字体   
mpl.rcParams['axes.unicode_minus'] = False #解决保存图像是负号'-'显示为方块的问题

# venn2和venn2_circles接受一个3元素（Ab，aB，AB）构成的 tuple 作为各个子集所包含元素的个数（不是具体的元素）：
#
# Ab：包含A，但不包含B，即A中非B的部分，A∩¬B
# aB：包含B，但不包含A，即B中非A，B∩¬A
# AB：既包含A，又包含B，即A与B的交集，A∩B
# 类似地，venn3与venn3_circles 接受一个7个元素构成的元组作为各个子集的大小（Abc, aBc, ABc, abC, AbC, aBC, ABC）


fig, ax = plt.subplots(figsize=(10, 10))
v = venn3(subsets = (10, 10, 4, 10, 4, 4, 2), set_labels = ('A', 'B', 'C'), ax=ax)
v.get_label_by_id('100').set_text('Executive')
v.get_label_by_id('010').set_text('Legislative')
v.get_label_by_id('001').set_text('Judicial')
v.get_label_by_id('110').set_text('Example 1')
v.get_label_by_id('011').set_text('Example 2')
v.get_label_by_id('101').set_text('Example 3')
v.get_label_by_id('111').set_text('---')
plt.title("The Three Branches of the US Government")
example_text = ('Example 1: The Vice President is considered "President of the Senate" and can vote to break ties.\n'
                'Example 2: The Legislature confirms Supreme Court justices.\n'
                'Example 3: The Executive appoints potential Supreme Court justices.')
text = fig.text(0.0, 0.05, example_text, ha='left', va='bottom', size=14)
text.set_path_effects([path_effects.Normal()])
plt.show()


fig, ax = plt.subplots(figsize=(10, 10))
v = venn2(subsets = (1,2,2,1), set_labels = ('A', 'B'), ax=ax)
v.get_label_by_id('01').set_text('AAA、AA、A')
v.get_label_by_id('10').set_text('BBB、BB、B')
v.get_label_by_id('11').set_text('AB、AAB')
text = fig.text(0.0, 0.05, example_text, ha='left', va='bottom', size=14)
text.set_path_effects([path_effects.Normal()])
plt.show()

fig, ax = plt.subplots(figsize=(10, 10))
v = venn2(subsets ={'10': 1, '01': 1, '11': 1}, set_labels = ('取现有积分', '取现无积分'), ax=ax)
# subsets, 既可以是一个列表，也可以是一个字典，或者是一个元祖；其值分别表示各自子集的大小，而非集合中的元素内容；
# v = venn2(subsets=[set([1,2,3]), set([1])], set_labels=('A', 'B')) # 构造一种包含关系
# v = venn2(subsets=(3, 0, 2), set_labels=('A', 'B')) # 构造一种包含关系

v.get_label_by_id('01').set_text('''邮储、上海、
浦发、交行、
农行、北京、
华夏、工商、
平安、广发、
建行、招行、
民生''')
v.get_label_by_id('10').set_text('''中信、光大、
兴业''')
v.get_label_by_id('11').set_text('''中行
(中银系列有积分
长城卡无积分)''')
v.get_label_by_id('01').set_multialignment('left')
v.get_label_by_id('10').set_multialignment('left')
v.get_label_by_id('01').set_fontsize(22)
v.get_label_by_id('10').set_fontsize(22)
v.get_label_by_id('11').set_fontsize(22)
# text = fig.text(0.0, 0.05, example_text, ha='left', va='bottom', size=14)
# text.set_path_effects([path_effects.Normal()])
plt.show()

def main():
    pass


if __name__ == '__main__':
    main()