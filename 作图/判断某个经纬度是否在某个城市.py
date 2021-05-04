#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 第一步，获取城市的行政区域边界对应的经纬度坐标
# 第二步，用matplotlib.path，确定某个点的坐标是否在某个多边形范围内

# 深圳 113.7964, 22.7856
# 广州 113.5107, 23.2196

import json


# 数据来源：https://github.com/longwosion/geojson-map-china.git
with open(r'D:\Users\gswyhq\github_project\geojson-map-china\geometryProvince\44.json', 'r', encoding='utf-8')as f:
    data = json.load(f)

# 广州外多边形的经纬度点
gz_coordinates= [d['geometry']['coordinates'] for d in data['features'] if d['properties'].get('name')=='广州市'][0][0]

def test1():
    from matplotlib.path import Path
    path = Path(gz_coordinates, closed=True)

    ret = path.contains_points([(113.7964, 22.7856), (113.5107, 23.2196)])
    print(ret)
    # [False  True]

def test2():
    import numpy as np
    from shapely.geometry import Point
    from shapely.geometry.polygon import Polygon
    buffer = np.array(gz_coordinates).reshape(len(gz_coordinates), 2)
    for lng, lat in [(113.7964, 22.7856), (113.5107, 23.2196)]:
        point = Point(lng, lat)
        ret = Polygon(buffer).contains(point)
        print(ret)

def main():
    test1()
    test2()



if __name__ == '__main__':
    main()