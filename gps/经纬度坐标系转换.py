#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import os 

DOUBLE_X_PI = 3.14159265358979324 * 3000.0 / 180.0
DOUBLE_PI = 3.1415926535897932384626  # π
DOUBLE_A = 6378245.0  # 长半轴
DOUBLE_EE = 0.00669342162296594323  # 偏心率平方

USERNAME = os.getenv('USERNAME')

#     # * 百度坐标系 (BD-09) 与 火星坐标系 (GCJ-02)的转换
#     # * 火星坐标系 (GCJ-02) 与百度坐标系 (BD-09) 的转换
#     # * 即谷歌、高德 转 百度
#     国内各地图API坐标系统比较 API	坐标系
#     百度地图API	百度坐标
#     腾讯搜搜地图API	火星坐标
#     搜狐搜狗地图API	搜狗坐标
#     阿里云地图API	火星坐标
#     图吧MapBar地图API	图吧坐标
#     高德MapABC地图API	火星坐标
#     灵图51ditu地图API	火星坐标
# WGS84：为一种大地坐标系，也是目前广泛使用的GPS全球卫星定位系统使用的坐标系。
# GCJ02：是由中国国家测绘局制订的地理信息系统的坐标系统。由WGS84坐标系经加密后的坐标系。
# BD09：为百度坐标系，在GCJ02坐标系基础上再次加密。其中bd09ll表示百度经纬度坐标，bd09mc表示百度墨卡托米制坐标

def gcj02_to_bd09(lng, lat):
    """
    火星坐标系(GCJ-02)转百度坐标系(BD-09)
    谷歌、高德——>百度
    :param lng:火星坐标经度
    :param lat:火星坐标纬度
    :return:
    """
    if isinstance(lng, str):
        lng = float(lng)
    if isinstance(lat, str):
        lat = float(lat)
    z = math.sqrt(lng * lng + lat * lat) + 0.00002 * math.sin(lat * DOUBLE_X_PI)
    theta = math.atan2(lat, lng) + 0.000003 * math.cos(lng * DOUBLE_X_PI)
    bd_lng = z * math.cos(theta) + 0.0065
    bd_lat = z * math.sin(theta) + 0.006
    return [bd_lng, bd_lat]


def bd09_to_gcj02(bd_lon, bd_lat):
    """
    百度坐标系(BD-09)转火星坐标系(GCJ-02)
    百度——>谷歌、高德
    :param bd_lat:百度坐标纬度
    :param bd_lon:百度坐标经度
    :return:转换后的坐标列表形式
    """

    if isinstance(bd_lon, str):
        bd_lon = float(bd_lon)
    if isinstance(bd_lat, str):
        bd_lat = float(bd_lat)
    x = bd_lon - 0.0065
    y = bd_lat - 0.006
    z = math.sqrt(x * x + y * y) - 0.00002 * math.sin(y * DOUBLE_X_PI)
    theta = math.atan2(y, x) - 0.000003 * math.cos(x * DOUBLE_X_PI)
    gg_lng = z * math.cos(theta)
    gg_lat = z * math.sin(theta)
    return [gg_lng, gg_lat]


def wgs84_to_gcj02(lng, lat):
    """
    WGS84转GCJ02(火星坐标系)
    :param lng:WGS84坐标系的经度
    :param lat:WGS84坐标系的纬度
    :return:
    """
    if isinstance(lng, str):
        lng = float(lng)
    if isinstance(lat, str):
        lat = float(lat)
    if out_of_china(lng, lat):  # 判断是否在国内
        return [lng, lat]
    dlat = _transformlat(lng - 105.0, lat - 35.0)
    dlng = _transformlng(lng - 105.0, lat - 35.0)
    radlat = lat / 180.0 * DOUBLE_PI
    magic = math.sin(radlat)
    magic = 1 - DOUBLE_EE * magic * magic
    sqrtmagic = math.sqrt(magic)
    dlat = (dlat * 180.0) / ((DOUBLE_A * (1 - DOUBLE_EE)) / (magic * sqrtmagic) * DOUBLE_PI)
    dlng = (dlng * 180.0) / (DOUBLE_A / sqrtmagic * math.cos(radlat) * DOUBLE_PI)
    mglat = lat + dlat
    mglng = lng + dlng
    return [mglng, mglat]


def gcj02_to_wgs84(lng, lat):
    """
    GCJ02(火星坐标系)转GPS84
    :param lng:火星坐标系的经度
    :param lat:火星坐标系纬度
    :return:
    """
    if isinstance(lng, str):
        lng = float(lng)
    if isinstance(lat, str):
        lat = float(lat)
    if out_of_china(lng, lat):
        return [lng, lat]
    dlat = _transformlat(lng - 105.0, lat - 35.0)
    dlng = _transformlng(lng - 105.0, lat - 35.0)
    radlat = lat / 180.0 * DOUBLE_PI
    magic = math.sin(radlat)
    magic = 1 - DOUBLE_EE * magic * magic
    sqrtmagic = math.sqrt(magic)
    dlat = (dlat * 180.0) / ((DOUBLE_A * (1 - DOUBLE_EE)) / (magic * sqrtmagic) * DOUBLE_PI)
    dlng = (dlng * 180.0) / (DOUBLE_A / sqrtmagic * math.cos(radlat) * DOUBLE_PI)
    mglat = lat + dlat
    mglng = lng + dlng
    return [lng * 2 - mglng, lat * 2 - mglat]


def bd09_to_wgs84(bd_lon, bd_lat):
    """百度转GPS"""

    lon, lat = bd09_to_gcj02(bd_lon, bd_lat)
    return gcj02_to_wgs84(lon, lat)


def wgs84_to_bd09(lon, lat):
    """GPS转百度"""
    lon, lat = wgs84_to_gcj02(lon, lat)
    return gcj02_to_bd09(lon, lat)

def _transformlat(lng, lat):
    ret = -100.0 + 2.0 * lng + 3.0 * lat + 0.2 * lat * lat + \
          0.1 * lng * lat + 0.2 * math.sqrt(math.fabs(lng))
    ret += (20.0 * math.sin(6.0 * lng * DOUBLE_PI) + 20.0 *
            math.sin(2.0 * lng * DOUBLE_PI)) * 2.0 / 3.0
    ret += (20.0 * math.sin(lat * DOUBLE_PI) + 40.0 *
            math.sin(lat / 3.0 * DOUBLE_PI)) * 2.0 / 3.0
    ret += (160.0 * math.sin(lat / 12.0 * DOUBLE_PI) + 320 *
            math.sin(lat * DOUBLE_PI / 30.0)) * 2.0 / 3.0
    return ret


def _transformlng(lng, lat):
    ret = 300.0 + lng + 2.0 * lat + 0.1 * lng * lng + \
          0.1 * lng * lat + 0.1 * math.sqrt(math.fabs(lng))
    ret += (20.0 * math.sin(6.0 * lng * DOUBLE_PI) + 20.0 *
            math.sin(2.0 * lng * DOUBLE_PI)) * 2.0 / 3.0
    ret += (20.0 * math.sin(lng * DOUBLE_PI) + 40.0 *
            math.sin(lng / 3.0 * DOUBLE_PI)) * 2.0 / 3.0
    ret += (150.0 * math.sin(lng / 12.0 * DOUBLE_PI) + 300.0 *
            math.sin(lng / 30.0 * DOUBLE_PI)) * 2.0 / 3.0
    return ret


def out_of_china(lng, lat):
    """
    判断是否在国内，不在国内不做偏移
    :param lng:
    :param lat:
    :return:
    """
    return not (lng > 73.66 and lng < 135.05 and lat > 3.86 and lat < 53.55)

def generator_test_sql():
    ts = [['121.6247', '38.9217'], ['121.526047', '29.800207'], ['114.052895', '22.613007'], ['116.5513', '37.3948'], ['122.759', '40.8597'], ['120.4917', '36.1566'], ['106.6729', '26.6279'], ['103.818764', '36.09758'], ['114.9221', '37.6191'], ['103.8339', '36.0601'], ['116.249313', '37.694057'], ['118.628', '24.8907'], ['103.723724', '29.608976'], ['121.622818', '29.86599'], ['121.2385', '31.2367'], ['116.5785', '33.2595'], ['116.5408', '37.8977'], ['104.0975', '30.6975'], ['114.0283', '33.3742'], ['104.5922', '35.6442'], ['113.7802', '22.7574'], ['117.648', '36.9443'], ['122.9288', '39.7234'], ['114.4253', '23.1233'], ['114.996201', '30.092119'], ['121.044731', '31.517694'], ['112.91111', '33.73756'], ['117.2925', '31.8396'], ['103.4072', '23.3614'], ['109.7613', '38.2959'], ['104.1522', '30.8035'], ['119.0438', '33.6053'], ['118.2813', '34.9842'], ['113.6963', '35.1308'], ['116.0522', '36.465'], ['114.5262', '32.1916'], ['120.4338', '36.0693'], ['119.230659', '26.010643'], ['105.0511', '29.611'], ['121.1722', '31.1562'], ['116.0524', '36.4654'], ['116.361908', '40.042847'], ['103.8791', '30.6808'], ['110.7469', '27.111'], ['106.1946', '37.9999'], ['121.6357', '42.026'], ['129.5084', '42.9136'], ['113.4444', '35.373'], ['110.36792', '20.013502'], ['110.3881', '21.2616'], ['103.739822', '29.539621'], ['113.7216', '22.7786'], ['114.496552', '37.994709'], ['114.9236', '27.8385'], ['117.8906', '39.3645'], ['109.867729', '40.662003'], ['113.4022', '22.5515'], ['116.120346', '24.291313'], ['121.185944', '31.884764'], ['109.12', '22.3349'], ['114.4002', '30.5175'], ['116.9807', '36.6603'], ['120.4239', '30.7077'], ['122.2793', '43.6322'], ['120.4557', '36.265'], ['113.2021', '33.7728'], ['121.4851', '37.4483'], ['126.0397', '44.9902'], ['116.1016', '24.2806'], ['104.0277', '35.9508'], ['120.482101', '30.088711'], ['123.2424', '41.2717'], ['113.7854', '22.803'], ['114.269836', '30.62174'], ['113.3645', '31.7179'], ['117.2264', '36.7101'], ['123.7935', '41.3089'], ['112.8034', '28.2562'], ['117.278709', '39.119072'], ['110.4734', '35.4755'], ['106.2408', '28.3145'], ['117.6936', '24.5069'], ['123.7935', '41.3089'], ['114.3847', '32.6101'], ['104.5538', '30.416'], ['122.2714', '43.6211'], ['120.902496', '31.974745'], ['118.7453', '32.2087'], ['109.4756', '36.5363'], ['119.3116', '39.8782'], ['123.3981', '41.6819'], ['118.8057', '32.1863'], ['112.208641', '31.035969'], ['118.5669', '35.8028'], ['121.5409', '29.9252'], ['120.8317', '37.3115'], ['105.5962', '30.5384'], ['120.4896', '36.1528'], ['118.7439', '30.2646'], ['123.476021', '41.769817']]
    lon_lat2 = []
    for bd_lon, bd_lat in ts:
        lon, lat = bd09_to_wgs84(bd_lon, bd_lat)
        lon_lat2.append([lon, lat])
        print('''select * from test_db.addr_clean_poi_2021
    LATERAL VIEW Distance('{}', '{}', lng, lat) t2 as `distance`
    where substr(lng, 0, 6)='{}' and  substr(lat, 0, 5)='{}'
    order by t2.distance
    limit 1;'''.format(lon, lat, str(lon)[:6], str(lat)[:5]))
        print('-'*80)
    print(    '''select * from test_db.addr_clean_poi_2021
    where substr(lng, 0, 6) in {} and  substr(lat, 0, 5) in {}
    '''.format(tuple([str(t[0])[:6] for t in lon_lat2]), tuple([str(t[1])[:5] for t in lon_lat2])))

def main():
    # lng_lat = '120.885657,32.024312'
    lng_lat = '121.650069,31.813379'
    lng_lat = '121.650069,31.813379'
    lng_lat = '125.70586,44.549753'
    lng_lat = '113.963626,22.544535'
    # lng_lat = '113.82,22.73'
    lng_lat = '113.953,22.542'
    lng = lng_lat.split(',')[0]
    lat = lng_lat.split(',')[1]
    print('原始，', lng_lat)
    print("百度", wgs84_to_bd09(lng, lat))
    bd_lng, bd_lat = wgs84_to_bd09(lng, lat)
    print(f'百度保留6位小数：{bd_lng:.6f},{bd_lat:.6f}')
    gcj02_lng, gcj02_lat = bd09_to_gcj02(*wgs84_to_bd09(lng, lat))
    print("高德", gcj02_lng, gcj02_lat)

    print('WGS84-> bd09-> WGS84', bd09_to_wgs84(bd_lng, bd_lat))

    wgs84_lng, wgs84_lat = gcj02_to_wgs84(gcj02_lng, gcj02_lat)
    print('WGS84', wgs84_lng, wgs84_lat )

    print("{}, {}\n bd09->gcj02->bd09: {}".format(bd_lng, bd_lat, gcj02_to_bd09(*bd09_to_gcj02(bd_lng, bd_lat))))

    print("{}, {}\n wgs84->gcj02->wgs84: {}".format(lng, lat, gcj02_to_wgs84(*wgs84_to_gcj02(lng, lat))))

    print("{}, {}\n gcj02->wgs84: {}".format(lng, lat, gcj02_to_wgs84(lng, lat)))
    print("{}, {}\n bd09->wgs84: {}".format(lng, lat, bd09_to_wgs84(lng, lat)))
    print("{}, {}\n wgs84->gcj02: {}".format(lng, lat, wgs84_to_gcj02(lng, lat)))
    print("{}, {}\n bd09->gcj02: {}".format(lng, lat, bd09_to_gcj02(lng, lat)))
    print("{}, {}\n gcj02->bd09: {}".format(lng, lat, gcj02_to_bd09(lng, lat)))
    print("{}, {}\n wgs84->bd09: {}".format(lng, lat, wgs84_to_bd09(lng, lat)))
    print("{:.6f}".format(1 / 6))

# gcj02towgs84__3(113.963626, 22.544535)
# Out[25]: [113.9587163595357, 22.54750594161637]
# gcj02_to_wgs84(113.963626, 22.544535)
# Out[26]: [113.9587163595357, 22.54750594161637]
# GCJ02: 113.963626, 22.544535
# BD09: 113.97013394848933, 22.550420192703207
# wgs84:

# 坐标系转换
# 国测局规定：互联网地图在国内必须至少使用 GCJ02 进行首次加密，不允许直接使用 WGS84 坐标下的地理数据，同时任何坐标系均不可转换为 WGS84 坐标。因此不存在将 GCJ-02 坐标转换为 WGS84 坐标的官方转换方法。
# 故而 百度坐标转换为高德是可以转换回来的；若百度坐标转换为WGS84,是转换不回来百度坐标的；


if __name__ == '__main__':
    main()

