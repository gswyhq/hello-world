#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pyecharts import options as opts
from pyecharts.charts import Map
from pyecharts.faker import Faker
from pyecharts.render import make_snapshot
from snapshot_selenium import snapshot

c = (
    Map()
    .add("商家A", [list(z) for z in zip(['广东', '北京', '上海', '江西', '湖南', '浙江', '江苏', '湖北'], [71, 42, 93, 88, 33, 132, 26, 120])], 
         "china",)
    .set_series_opts(label_opts=opts.LabelOpts(is_show=True, formatter="{b}:{c}")) # is_show=True 是否显示标签
    .set_global_opts(title_opts=opts.TitleOpts(title="Map-基本示例"))
)
# c.render(r"map_base.html")

image_file = r'map_base.png'
make_snapshot(snapshot, c.render(),
              image_file)
print(image_file)

def main():
    pass


if __name__ == '__main__':
    main()