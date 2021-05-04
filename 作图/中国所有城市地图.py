#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pyecharts import options as opts
from pyecharts.charts import Map
from pyecharts.faker import Faker
from pyecharts.render import make_snapshot
from snapshot_selenium import snapshot

c = (
    Map()
    .add(
        "商家A",
        [list(z) for z in zip(Faker.guangdong_city, Faker.values())],
        "china-cities",
        label_opts=opts.LabelOpts(is_show=True, formatter="{b}:{c}"), # is_show=True 是否显示标签
    )
    .set_global_opts(
        title_opts=opts.TitleOpts(title="Map-中国地图（带城市）"),
        visualmap_opts=opts.VisualMapOpts(),
    )
    # .render("map_china_cities.html")
)

image_file = r'china_cities.png'
make_snapshot(snapshot, c.render(),
              image_file)
print(image_file)

def main():
    pass


if __name__ == '__main__':
    main()