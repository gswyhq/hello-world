#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
from pyecharts import options as opts
from pyecharts.charts import Polar
from pyecharts import options as opts
from pyecharts.charts import Polar
from pyecharts.faker import Faker
from pyecharts.commons.utils import JsCode

# https://gallery.pyecharts.org/#/Polar/polar_effectscatter
# 关键词A和关键词B的相关度 = 同时包含关键词A和关键词B的文章的阅读数 / 包含关键词A的文章的阅读数
# 气泡面积代表热度指数值大小。每个关键词的气泡大小代表该关键词在分析时间段内的热度大小，跟被浏览、阅读的数量呈正相关；
# 颜色代表该关键词的热度趋势。绿色代表呈下降趋势，红色代表呈上升趋势；
# 圆环深浅代表相关度大小。从里到外有三个圆环，由深到浅表明该词跟核心词（也就是“奥迪A6”）相关关系的由强变弱。
#

c = (
    Polar(init_opts=opts.InitOpts(width="1600px", height="800px"))
    .add_schema(angleaxis_opts=opts.AngleAxisOpts(type_="category",
                                                axisline_opts=opts.AxisLineOpts(is_show=False, is_on_zero=False),
                                                axistick_opts=opts.AxisTickOpts(is_show=False),
                                                axislabel_opts=opts.LabelOpts(is_show=False),  # 隐藏Y轴
                                                  ),
                radiusaxis_opts=opts.RadiusAxisOpts(
                                                axisline_opts=opts.AxisLineOpts(is_show=False, is_on_zero=False),
                                                axistick_opts=opts.AxisTickOpts(is_show=False),
                                                axislabel_opts=opts.LabelOpts(is_show=False),  # 隐藏X轴
                                                splitline_opts=opts.SplitLineOpts(is_show=True),
                )
                )
    # value 值越小，离圆心越近；symbolSize值越大，点的圆圈越大；
    .add("A类", [{'value': 0, 'symbolSize': 60, 'name': '奥迪a6'},
         {'value': 8, 'symbolSize': 29, 'name': '宝马'},
         {'value': 9, 'symbolSize': 20, 'name': '奔驰'},
         {'value': 8, 'symbolSize': 37, 'name': '宝马5系'},
         {'value': 10, 'symbolSize': 42, 'name': '奥迪'},
         {'value': 12, 'symbolSize': 55, 'name': '汽车'},
         {'value': 13, 'symbolSize': 24, 'name': '4s店'},
         {'value': 6, 'symbolSize': 22, 'name': '豪华车'},
         {'value': 12, 'symbolSize': 21, 'name': '沃尔沃'},
         {'value': 14, 'symbolSize': 34, 'name': '雷克萨斯'},
         {'value': 11, 'symbolSize': 25, 'name': '帕萨特', "seriesIndex":'', "xAxisIndex": '', "geoIndex": "", "yAxisIndex": ''},
         {'value': 15, 'symbolSize': 39, 'name': '路虎'},
         {'value': 8, 'symbolSize': 42, 'name': '豪华轿车'},], type_="scatter",
         itemstyle_opts = opts.ItemStyleOpts(color='#77ee00',), # 圆圈的颜色；
         label_opts=opts.LabelOpts(
             is_show=True,
             position = 'middle',
             color='#229933', # 字的颜色；
             formatter=JsCode(
                 "function(params){return params.name +' : '+ params.value;}"
             )
         ),
         )
    .add("B类", [
         {'value': 16, 'symbolSize': 27, 'name': '凯迪拉克'},
         {},
         {'value': 12, 'symbolSize': 35, 'name': '宾利'},
         {},
         {'value': 17.5, 'symbolSize': 49, 'name': '英菲尼迪'}
    ],
         type_="scatter",
         itemstyle_opts = opts.ItemStyleOpts(color='#77eeff', color0='#000000'),
         tooltip_opts=opts.TooltipOpts(is_show=False, is_show_content=False, padding=0),
         areastyle_opts = opts.AreaStyleOpts(color='#ffffff'),
         effect_opts = opts.EffectOpts(is_show=False, color='#000000'),
         label_opts=opts.LabelOpts(
             is_show=True,
             position = 'middle',
             color='#2244ff',
             formatter=JsCode(
                 "function(params){return params.name +' : '+ params.value;}"
             )
         )
         )
    .set_global_opts(title_opts=opts.TitleOpts(title="Polar-EffectScatter"),
                     xaxis_opts=opts.AxisOpts(is_show=False),
                     yaxis_opts=opts.AxisOpts(is_show=False),
                     legend_opts=opts.LegendOpts(is_show=True),
                     # visualmap_opts=[opts.VisualMapOpts(type_="size", max_=15, min_=0, is_show=False,
                     #                                    range_size=[20, 50]
                     #                                    ),
                     #                 # opts.VisualMapOpts(
                     #                 #     type_="color", max_=18, min_=0, dimension=1
                     #                 # )
                     #                 ]
                     )
    .render(r"D:\Users\Downloads\polar_effectscatter.html")
)

def main():
    pass


if __name__ == '__main__':
    main()