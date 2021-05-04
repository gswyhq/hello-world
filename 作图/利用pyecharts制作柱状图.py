#!/usr/bin/python3
# coding: utf-8

自从 v0.3.2 开始，为了缩减项目本身的体积以及维持 pyecharts 项目的轻量化运行，pyecharts 将不再自带地图 js 文件。如用户需要用到地图图表，可自行安装对应的地图文件包。下面介绍如何安装。

全球国家地图: echarts-countries-pypkg (1.9MB): 世界地图和 213 个国家，包括中国地图
中国省级地图: echarts-china-provinces-pypkg (730KB)：23 个省，5 个自治区
中国市级地图: echarts-china-cities-pypkg (3.8MB)：370 个中国城市
中国县区级地图: echarts-china-counties-pypkg (4.1MB)：2882 个中国县·区
中国区域地图: echarts-china-misc-pypkg (148KB)：11 个中国区域地图，比如华南、华北。
选择自己需要的安装

pip3 install echarts-countries-pypkg
pip3 install echarts-china-provinces-pypkg
pip3 install echarts-china-cities-pypkg
pip3 install echarts-china-counties-pypkg
pip3 install echarts-china-misc-pypkg
pip3 install echarts-united-kingdom-pypkg

from pyecharts.charts import Bar
from pyecharts import options as opts

# 生成 HTML
bar = (
    Bar()
    .add_xaxis(["衬衫", "毛衣", "领带", "裤子", "风衣", "高跟鞋", "袜子"])
    .add_yaxis("商家A", [114, 55, 27, 101, 125, 27, 105])
    .add_yaxis("商家B", [57, 134, 137, 129, 145, 60, 49])
    .set_global_opts(title_opts=opts.TitleOpts(title="某商场销售情况"))
)
bar.render(path="render2.html")

# 生成图片
from snapshot_selenium import snapshot as driver

from pyecharts import options as opts
from pyecharts.charts import Bar
from pyecharts.render import make_snapshot


def bar_chart() -> Bar:
    c = (
        Bar()
        .add_xaxis(["衬衫", "毛衣", "领带", "裤子", "风衣", "高跟鞋", "袜子"])
        .add_yaxis("商家A", [114, 55, 27, 101, 125, 27, 105])
        .add_yaxis("商家B", [57, 134, 137, 129, 145, 60, 49])
        .reversal_axis()
        .set_series_opts(label_opts=opts.LabelOpts(position="right"))
        .set_global_opts(title_opts=opts.TitleOpts(title="Bar-测试渲染图片"))
    )
    return c

# 需要安装 snapshot_selenium; 当然也可以使用： snapshot_phantomjs
make_snapshot(driver, bar_chart().render(path="render2.html"), "bar.png")

def main():
    pass


if __name__ == '__main__':
    main()
