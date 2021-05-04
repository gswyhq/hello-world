#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pyecharts.charts import Pie
from pyecharts import options as opts

from pyecharts.render import make_snapshot
from snapshot_selenium import snapshot

attr = ["衬衫", "羊毛衫", "雪纺衫", "裤子", "高跟鞋", "袜子"]
v1 = [11, 12, 13, 10, 10, 10]
pie = Pie()
pie.add(
    series_name="饼图示例",
    data_pair=[[k,v ]for k, v in zip(attr, v1)],
    # is_label_show=True,
    # is_more_utils=True
).set_series_opts(label_opts=opts.LabelOpts(formatter="{b}: {c}")).set_global_opts(
                title_opts=opts.TitleOpts(
                    title='name', title_textstyle_opts=opts.TextStyleOpts(font_size=23)
                ),
                tooltip_opts=opts.TooltipOpts(is_show=True),
            )

image_file = r'D:\Users\abcd\data\card\pos_neg\attr.png'
make_snapshot(snapshot, pie.render(),
                  image_file)


def main():
    pass


if __name__ == '__main__':
    main()