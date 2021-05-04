#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#导入需要的包
from pyecharts import options as opts
from pyecharts.charts import Sankey

#列全涉及的节点名称
nodes = [
    {"name": "产品1"},
    {"name": "产品2"},
    {"name": "产品3"},
    {"name": "产品4"},
    {"name": "产品5"},
    {"name": "产品6"},
    {"name": "新增"},
]

#节点之间的关系和数量，source起点，target终点，value数量
links = [
{"source": "新增", "target": "产品1", "value": 30},
{"source": "新增", "target": "产品5", "value": 30},
{"source": "产品2", "target": "产品5", "value": 30},
    {"source": "产品1", "target": "产品2", "value": 25},
    {"source": "产品2", "target": "产品3", "value": 20},
    {"source": "产品3", "target": "产品4", "value": 15},
    {"source": "产品5", "target": "产品6", "value": 10},
]
c = (
    Sankey(init_opts=opts.InitOpts(width="1200px", height="600px")) #设置图表的宽度和高度
    .add(
        "sankey",
        nodes,#读取节点
        links,#读取路径
        linestyle_opt=opts.LineStyleOpts(opacity=0.2, curve=0.5, color="source"),#设置线条样式
        label_opts=opts.LabelOpts(position="right"),#设置标签配置项
        node_align=str( "justify"),#设置节点对齐方式：right，left,justify(节点双端对齐)
    )
    .set_global_opts(title_opts=opts.TitleOpts(title="客户迁徙轨迹"))#表名
    .render(r"D:\Users\桑基图.html")#保存html文件
)

def main():
    pass


if __name__ == '__main__':
    main()