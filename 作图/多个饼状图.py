#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pyecharts import options as opts
from pyecharts.charts import Pie
from pyecharts.render import make_snapshot
from snapshot_selenium import snapshot
from pyecharts.commons.utils import JsCode


fn = """
    function(params) {
        if(params.name == '其他')
            return '\\n\\n\\n' + params.name + ' : ' + params.value + '%';
        return params.name + ' : ' + params.value + '%';
    }
    """


data_pair1 = [('卡面', 10677), ('颜值', 2754), ('卡面好看', 713), ('颜值不错', 660), ('换卡面', 591), ('颜值高', 503), ('钻石卡面', 382), ('颜值在线', 359), ('新卡面', 348), ('英菲尼迪卡面', 347)]
data_pair2 = [('好看', 448), ('高', 224), ('漂亮', 221), ('正好', 169), ('更好看', 164), ('太高', 152), ('正常', 134), ('设计', 134), ('还不错', 126), ('丑', 124)]

def new_label_opts():
    return opts.LabelOpts(formatter=JsCode(fn))

pie = Pie()
pie.add(
        "",
        data_pair1,
        center=["25%", "50%"],
        radius=[60, 80],
        label_opts=opts.LabelOpts(formatter="{b}: {c}（条）", is_show=True)
    )\
    .add(
        "",
        data_pair2,
        center=["75%", "50%"],
        radius=[60, 80],
        label_opts=new_label_opts()
    ).set_global_opts(
                title_opts=opts.TitleOpts(
                    title='卡面', title_textstyle_opts=opts.TextStyleOpts(font_size=23)
                ),
                tooltip_opts=opts.TooltipOpts(is_show=True),
                legend_opts = opts.LegendOpts(is_show=False),
            )

image_file = r'D:\Users\abcd\data\card\result18_20200730\多个饼状图.png'
make_snapshot(snapshot, pie.render(),
              image_file)

def main():
    pass


if __name__ == '__main__':
    main()