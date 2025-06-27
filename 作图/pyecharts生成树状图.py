#!/usr/bin/env python
# coding=utf-8

import os
from pyecharts import options as opts
from pyecharts import types
from pyecharts.charts import Tree
USERNAME = os.getenv("USERNAME")

# 数据结构和配置转换
data = {
    "name": "flare",
    "children": [
        {
            "name": "flex",
            "children": [{"name": "FlareVis", "value": 4116}]
        },
        {
            "name": "scale",
            "children": [
                {
                    "value": 1548,
                    "name": "CityE",
                    "label": opts.LabelOpts(
                        formatter= "\n".join([
                            '{title|{b}}{abg|}',
                            '  {weatherHead|Weather}{valueHead|Days}{rateHead|Percent}',
                            '{hr|}',
                            '  {Sunny|}{value|202}{rate|55.3%}',
                            '  {Cloudy|}{value|142}{rate|38.9%}',
                            '  {Showers|}{value|21}{rate|5.8%}'
                        ]),
                        background_color= '#eee',
                        border_color= '#777',
                        border_width= 1,
                        border_radius= 4,
                        rich= {
                            "title": {
                                "color": '#eee',
                                "align": 'center'
                            },
                            "abg": {
                                "backgroundColor": '#333',
                                "width": '100%',
                                "align": 'right',
                                "height": 25,
                                "borderRadius": [4, 4, 0, 0]
                            },
                            "Sunny": {
                                "height": 30,
                                "align": 'left',
                                "backgroundColor": {
                                    "image": "https://echarts.apache.org/examples/data/asset/img/weather/sunny_128.png"
                                }
                            },
                            "Cloudy": {
                                "height": 30,
                                "align": 'left',
                                "backgroundColor": {
                                    "image": "https://echarts.apache.org/examples/data/asset/img/weather/cloudy_128.png"
                                }
                            },
                            "Showers": {
                                "height": 30,
                                "align": 'left',
                                "backgroundColor": {
                                    "image": "https://echarts.apache.org/examples/data/asset/img/weather/showers_128.png"
                                }
                            },
                            "weatherHead": {
                                "color": '#333',
                                "height": 24,
                                "align": 'left'
                            },
                            "hr": {
                                "borderColor": '#777',
                                "width": '100%',
                                "borderWidth": 0.5,
                                "height": 0
                            },
                            "value": {
                                "width": 20,
                                "padding": [0, 20, 0, 30],
                                "align": 'left'
                            },
                            "valueHead": {
                                "color": '#333',
                                "width": 20,
                                "padding": [0, 20, 0, 30],
                                "align": 'center'
                            },
                            "rate": {
                                "width": 40,
                                "align": 'right',
                                "padding": [0, 10, 0, 0]
                            },
                            "rateHead": {
                                "color": '#333',
                                "width": 40,
                                "align": 'center',
                                "padding": [0, 10, 0, 0]
                            }
                        }
                    )
                },
                {"name": "LinearScale", "value": 1316},
                {"name": "这里被下面的多行遮挡住了", "value": 3151},
                {
                    "name": "这里是多行，本行内容遮挡住了其他行了\n第二行\n第三行\n第四行\n第5行\n第6行\n第7行\n第8行\n第9行",
                    "value": 1821,
                    "label": {  # 特殊文字配置
                        "color": "#FF4500",
                        "fontWeight": "bold",
                        "backgroundColor": "#FFFACD",
                        "fontSize": 14,  # 调小字号
                        "overflow": "break"  # 自动换行
                        },
                },
                {"name": "这里被上面多行遮挡住了", "value": 5833},
                {"name": "账上",
                 "itemStyle": {  # 单独配置该节点样式
                     "color": "#FF4500",
                     "borderColor": "#8B0000",
                     "shadowColor": "#FFA07A",
                     "shadowBlur": 10,
                     "height": 40  # 高度设置
                 },
                 "children": [
                    {"name": "节点1", "value": 5834},
                    {"name": "节点2", "value": 5835},
                ]}
            ]
        },
        {
            "name": "display",
            "children": [{"name": "DirtySprite", "value": 8833}]
        }
    ]
}

# 创建树图
tree = (
    Tree(init_opts=opts.InitOpts(width="1000px", height="1600px"))
    .add(
        series_name="综金渗透率",
        data=[data],
        orient='LR', # 横向树状图
        # edge_shape="polyline",  # 使用折线连接，减少交叉
        layout="orthogonal",  # 正交布局，更易控制
        symbol_size=10,
        label_opts=opts.LabelOpts(
            position="left",
            vertical_align="middle",
            horizontal_align="right",
            formatter="{b}",
            padding=[12, 15, 12, 15],  # 增加内边距
            font_size=12,
            distance=5, # 标签与节点的距离
        ),
        leaves_opts=opts.TreeLeavesOpts(
            label_opts=opts.LabelOpts(
                position="right",
                vertical_align="middle",
                horizontal_align="left",
                padding=[12, 15, 12, 15]  # 增加内边距
            ),
        ),
        is_expand_and_collapse=False,  # 关闭展开折叠
        emphasis_opts = opts.EmphasisOpts(focus='descendant'),
        pos_left="15%",
        pos_right="15%",
        pos_top="10%",
        pos_bottom="10%",
        initial_tree_depth=3,  # 默认展开层级
    )
    .set_global_opts(
        tooltip_opts=opts.TooltipOpts(
            trigger="item",
            trigger_on="mousemove",
            is_append_to_body=True  # 防止tooltip被遮挡
        ),
        legend_opts=opts.LegendOpts(
            pos_left="3%",
            pos_top="2%",
            orient="vertical",
            border_color="#c23531"
        )
    )
)

# 生成HTML文件
tree.render(rf"D:\Users\{USERNAME}\Downloads\tree_chart.html")

def main():
    pass


if __name__ == "__main__":
    main()

# 资料来源：https://echarts.apache.org/examples/zh/index.html#chart-type-rich


