#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from datetime import datetime
from pyecharts.charts import Geo
from pyecharts import options as opts
from pyecharts import types
from pyecharts.globals import GeoType, ChartType, BMapType
from pyecharts import options as opts
from pyecharts.charts import Map
from pyecharts.faker import Faker

def shengfen_color():
    data= [
              {"name": '北京', "is_selected": False, "value": 1},
              {"name": '天津', "is_selected": False, "value": 2},
              {"name": '上海', "is_selected": False, "value": 3},
              {"name": '重庆', "is_selected": False, "value": 4},
              {"name": '河北', "is_selected": False, "value": 5},
              {"name": '河南', "is_selected": False, "value": 6},
              {"name": '云南', "is_selected": False, "value": 7},
              {"name": '辽宁', "is_selected": False, "value": 8},
              {"name": '黑龙江', "is_selected": False, "value": 9},
              {"name": '湖南', "is_selected": False, "value": 10},
              {"name": '安徽', "is_selected": False, "value": 11},
              {"name": '山东', "is_selected": False, "value": 12},
              {"name": '新疆', "is_selected": False, "value": 13},
              {"name": '江苏', "is_selected": False, "value": 14},
              {"name": '浙江', "is_selected": False, "value": 15},
              {"name": '江西', "is_selected": False, "value": 16},
              {"name": '湖北', "is_selected": False, "value": 17},
              {"name": '广西', "is_selected": False, "value": 18},
              {"name": '甘肃', "is_selected": False, "value": 19},
              {"name": '山西', "is_selected": False, "value": 20},
              {"name": '内蒙古', "is_selected": False, "value": 21},
              {"name": '陕西', "is_selected": False, "value": 22},
              {"name": '吉林', "is_selected": False, "value": 23},
              {"name": '福建', "is_selected": False, "value": 24},
              {"name": '贵州', "is_selected": False, "value": 25},
              {"name": '广东', "is_selected": False, "value": 26},
              {"name": '青海', "is_selected": False, "value": 27},
              {"name": '西藏', "is_selected": False, "value": 28},
              {"name": '四川', "is_selected": False, "value": 29},
              {"name": '宁夏', "is_selected": False, "value": 30},
              {"name": '海南', "is_selected": False, "value": 31},
              {"name": '台湾', "is_selected": False, "value": 32},
              {"name": '香港', "is_selected": False, "value": 33},
              {"name": '澳门', "is_selected": False, "value": 34}
          ] # 各省地图颜色数据依赖"value"

    splitList= [{'start': 1, 'end': 1, 'label': '北京', 'color': '#DC143C'},
                 {'start': 2, 'end': 2, 'label': '天津', 'color': '#FF00FF'},
                 {'start': 3, 'end': 3, 'label': '上海', 'color': '#8B008B'},
                 {'start': 4, 'end': 4, 'label': '重庆', 'color': '#8A2BE2'},
                 {'start': 5, 'end': 5, 'label': '河北', 'color': '#9370DB'},
                 {'start': 6, 'end': 6, 'label': '河南', 'color': '#0000FF'},
                 {'start': 7, 'end': 7, 'label': '云南', 'color': '#6495ED'},
                 {'start': 8, 'end': 8, 'label': '辽宁', 'color': '#87CEEB'},
                 {'start': 9, 'end': 9, 'label': '黑龙江', 'color': '#00BFFF'},
                 {'start': 10, 'end': 10, 'label': '湖南', 'color': '#AFEEEE'},
                 {'start': 11, 'end': 11, 'label': '安徽', 'color': '#00FFFF'},
                 {'start': 12, 'end': 12, 'label': '山东', 'color': '#98FB98'},
                 {'start': 13, 'end': 13, 'label': '新疆', 'color': '#00FF00'},
                 {'start': 14, 'end': 14, 'label': '江苏', 'color': '#FAFAD2'},
                 {'start': 15, 'end': 15, 'label': '浙江', 'color': '#FFFF00'},
                 {'start': 16, 'end': 16, 'label': '江西', 'color': '#FFE4B5'},
                 {'start': 17, 'end': 17, 'label': '湖北', 'color': '#FFA500'},
                 {'start': 18, 'end': 18, 'label': '广西', 'color': '#F4A460'},
                 {'start': 19, 'end': 19, 'label': '甘肃', 'color': '#D2691E'},
                 {'start': 20, 'end': 20, 'label': '山西', 'color': '#FF4500'},
                 {'start': 21, 'end': 21, 'label': '内蒙古', 'color': '#CD5C5C'},
                 {'start': 22, 'end': 22, 'label': '陕西', 'color': '#FF0000'},
                 {'start': 23, 'end': 23, 'label': '吉林', 'color': '#fce8cd'},
                 {'start': 24, 'end': 24, 'label': '福建', 'color': '#fad8e8'},
                 {'start': 25, 'end': 25, 'label': '贵州', 'color': '#fad8e8'},
                 {'start': 26, 'end': 26, 'label': '广东', 'color': '#ddcfe8'},
                 {'start': 27, 'end': 27, 'label': '青海', 'color': '#fad8e9'},
                 {'start': 28, 'end': 28, 'label': '西藏', 'color': '#ddcfe6'},
                 {'start': 29, 'end': 29, 'label': '四川', 'color': '#e4f1d5'},
                 {'start': 30, 'end': 30, 'label': '宁夏', 'color': '#fefcd5'},
                 {'start': 31, 'end': 31, 'label': '海南', 'color': '#fad8e9'},
                 {'start': 32, 'end': 32, 'label': '台湾', 'color': '#fce8cd'},
                 {'start': 33, 'end': 33, 'label': '香港', 'color': '#dc9bbb'},
                 {'start': 34, 'end': 34, 'label': '澳门', 'color': '#E0F7CC'}]
    # range_color = ['#FFC0CB', '#DC143C', '#FF00FF', '#8B008B', '#8A2BE2', '#9370DB', '#0000FF', '#6495ED', '#87CEEB', '#00BFFF', '#AFEEEE', '#00FFFF', '#98FB98', '#00FF00', '#FAFAD2', '#FFFF00', '#FFE4B5', '#FFA500', '#F4A460', '#D2691E', '#FF4500', '#CD5C5C', '#FF0000']
    data_pair = [opts.MapItem(**d) for d in data]
    c = (
        Map()
            .add("", data_pair=data_pair,
                 label_opts = opts.LabelOpts(is_show=False),
                 maptype="china"
                 )
            .set_global_opts(
            legend_opts=opts.LegendOpts(type_="scroll", pos_left="50%", pos_top="20%", orient="vertical"),
            title_opts=opts.TitleOpts(title="Map-中国地图", pos_left="40%", pos_top="8%", ),
            # visualmap_opts=opts.VisualMapOpts(is_piecewise=False, pieces=pieces),
            visualmap_opts=opts.VisualMapOpts(textstyle_opts=opts.TextStyleOpts(color= "blue", font_size=12),
                                              is_piecewise=True,
                                              # pos_left='10%',
                                              # pos_top="0%",
                                                # pos_right="0%",
                                                # pos_bottom="0%",
                                              pieces=splitList),
        )
            .render(r'D:\Users\gswyhq\data\中国.html')
    )

def guangzhou():
    from pyecharts.charts import Map

    districts = ['白云区', '从化区', '番禺区', '海珠区', '花都区', '黄埔区', '荔湾区', '南沙区', '天河区', '越秀区', '增城区']
    value = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

    map = Map()
    map.add("",
            [(k, v) for k, v in zip(districts, value)], maptype='广州', ).set_global_opts(
        visualmap_opts=opts.VisualMapOpts(textstyle_opts=opts.TextStyleOpts(color="blue", font_size=12),
                                          min_=min(value), max_=max(value), split_number=len(set(value)),
                                          is_piecewise=False)
    )
    map.render(r'D:\Users\gswyhq\data\广州地图.html')

def main():
    shengfen_color()
    guangzhou()


if __name__ == '__main__':
    main()