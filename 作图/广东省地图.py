#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pyecharts import options as opts
from pyecharts.charts import Map
from pyecharts.faker import Faker
from pyecharts.render import make_snapshot
from snapshot_selenium import snapshot

# pyecharts.__version__
# Out[194]: '1.8.1'

c = (
    Map()
    .add("商家A", [list(z) for z in zip(Faker.guangdong_city, Faker.values())], "广东")
    .set_global_opts(
        title_opts=opts.TitleOpts(title="Map-广东地图"), visualmap_opts=opts.VisualMapOpts()
    )
)

c.render(r"map_guangdong.html")
image_file = r'map_guangdong.png'
make_snapshot(snapshot, c.render(),
              image_file)
print(image_file)


def main():
    pass


if __name__ == '__main__':
    main()