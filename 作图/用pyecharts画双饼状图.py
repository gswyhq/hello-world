#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pyecharts.charts import Pie
from pyecharts.commons.utils import JsCode
from pyecharts import options as opts
pie =Pie(init_opts=opts.InitOpts(width="800px", height="400px")) # title_pos='center'
pie.add("", [(k, v) for k, v in zip(['A', 'B', 'C', 'D', 'E', 'F'], [335, 321, 234, 135, 251, 148])], radius=[40, 55],label_opts = opts.LabelOpts(is_show=True, formatter="{b}: {c}\n({d}%)", ))
pie.add("", [(k, v) for k, v in zip(['H', 'I', 'J'], [335, 679, 204])], radius=[0, 30]).\
    set_global_opts(legend_opts=opts.LegendOpts(
        orient="vertical",
        pos_top="25%", # 设置图例上下位置；
        pos_left="20%"), # 设置图例左右位置
)

pie.render(r'D:\Users\data\card\积分\双饼状图.html')

def main():
    pass


if __name__ == '__main__':
    main()