#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pyecharts.options as opts
from pyecharts.charts import WordCloud

from pyecharts.render import make_snapshot
from snapshot_selenium import snapshot

"""
Gallery 使用 pyecharts 1.1.0
参考地址: https://gallery.echartsjs.com/editor.html?c=xS1jMxuOVm

http://gallery.pyecharts.org/#/WordCloud/basic_wordcloud

"""

data = [('机场贵宾', '1639888'),
 ('机场接送', '438000'),
 ('机场CIP', '995709'),
 ('酒店礼遇', '1376102'),
 ('运动健身', '377771'),
 ('航空延误险', '1936587'),
 ('导医服务', '387960'),
 ('机场停车', '1904364'),
 ('网购积分', '1516782'),
 ('多倍积分', '1917986'),
 ('洁牙/体检服务', '369598'),
 ('航空里程', '1140841'),
 ('免年费', '2423384'),
 ('办卡送礼', '326179'),
 ('加油返现', '107880'),
 ('道路救援', '423715'),
 ('失卡保障', '631481'),
 ('高端卡', '911982'),
 ('高尔夫服务', '183610'),
 ('超商优惠', '203312'),
 ('免货币转换费', '534227'),
 ('海淘返现', '249399'),
 ('全币种', '154675'),
 ('免息期长', '191173'),
 ('旅行意外险', '686867'),
 ('观影优惠', '188953'),
 ('消费返现', '570130'),
 ('取现优惠', '175910'),
 ('低利率', '177920'),
 ('境外消费', '324892'),
 ('全额提现', '54134'),
 ('免费全球Wifi', '81593'),
 ('美容/SPA服务', '1194'),
 ('卡面DIY设计', '323')]


c = (
    WordCloud()
    .add(series_name="信用卡标签", data_pair=data, word_size_range=[20, 66])  # word_size_range: 调节字体大小
    .set_global_opts(
        title_opts=opts.TitleOpts(
            title="信用卡标签", title_textstyle_opts=opts.TextStyleOpts(font_size=23)
        ),
        tooltip_opts=opts.TooltipOpts(is_show=True),
    )
)

# c.render("basic_wordcloud.html")

image_file = r'basic_wordcloud.png'
make_snapshot(snapshot, c.render(),
              image_file)
print(image_file)

def main():
    pass


if __name__ == '__main__':
    main()