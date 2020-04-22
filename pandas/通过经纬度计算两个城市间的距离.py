#!/usr/bin/python3
# coding: utf-8

import json
# import xlrd
from math import radians, cos, sin, asin, sqrt

# “经度”【longitude】、“纬度”【 Latitude】
# 数据来源于： https://github.com/pfinal/city
LONGITUDE_LATITUDE_FILE = '/home/gswyhq/github_projects/city/region.json'  # region.sql -> region.json
# CITY_CODE = '/home/gswyhq/github_projects/city/行政区划清单 V3.0 9.03.xlsx'

with open(LONGITUDE_LATITUDE_FILE)as f:
    data = json.load(f)
    LONGITUDE_LATITUDE_DATA = data['RECORDS']
    # 按地名长到短排序
    SORTED_LONGITUDE_LATITUDE_DATA = sorted(LONGITUDE_LATITUDE_DATA, key=lambda x: len(x.get('name', '')))

def get_long_lat(city):
    long_lat0 = None  # 精确匹配到；
    long_lat1 = None # 缺失末尾字
    long_lat2 = None # 末尾字不一致

    for data in SORTED_LONGITUDE_LATITUDE_DATA:
        name = data.get('name', '')
        lng = data['lng']
        lat = data['lat']
        if name == city:
            long_lat0 = lng, lat
            break
        elif name[:-1] == city:
            long_lat1 = lng, lat
        elif name[:-1] == city[:-1] and len(city) > 2:
            long_lat2 = lng, lat

    return long_lat0 or long_lat1 or long_lat2

def geodistance(lng1, lat1, lng2, lat2):
    """通过经纬度计算两点间的距离，单位：km"""
    #lng1,lat1,lng2,lat2 = (120.12802999999997,30.28708,115.86572000000001,28.7427)
    lng1, lat1, lng2, lat2 = map(radians, [float(lng1), float(lat1), float(lng2), float(lat2)]) # 经纬度转换成弧度
    dlon=lng2-lng1
    dlat=lat2-lat1
    a=sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    distance=2*asin(sqrt(a))*6371*1000 # 地球平均半径，6371km
    distance=round(distance/1000,3)
    return distance

def main():
    city1 = '浠水'
    city2 = '福田'
    lng1, lat1 = get_long_lat(city1)
    lng2, lat2 = get_long_lat(city2)
    distance = geodistance(lng1, lat1, lng2, lat2)
    print("`{}`到`{}`的距离是： {} km".format(city1, city2, distance))


if __name__ == '__main__':
    main()