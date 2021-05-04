#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pyecharts import options as opts
from pyecharts.charts import Map
from pyecharts.faker import Faker
from pyecharts.render import make_snapshot
from snapshot_selenium import snapshot

# 来源： https://gallery.pyecharts.org/#/Map/map_visualmap_piecewise?id=pyecharts-%E4%BB%A3%E7%A0%81-%E6%95%88%E6%9E%9C

c = (
    Map()
    .add("商家A", [list(z) for z in zip(Faker.provinces, Faker.values())], "china")
    .set_global_opts(
        title_opts=opts.TitleOpts(title="Map-VisualMap（分段型）"),
        visualmap_opts=opts.VisualMapOpts(max_=200, is_piecewise=True),
    )
)

c.render(r"map_visualmap_piecewise.html")
image_file = r'map_visualmap_piecewise.png'
make_snapshot(snapshot, c.render(),
              image_file)
print(image_file)

def main():
    pass


if __name__ == '__main__':
    main()