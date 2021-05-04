#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# pip install folium
# 特别说明，folium默认绘制的中国地图边界不对；需要更换底图
# https://blog.csdn.net/weixin_38169413/article/details/104806257

import folium

# 绘制世界地图
world_map = folium.Map()
world_map.save(r'世界地图.html')

# 绘制中国地图(该地图绘制的中国地图边界不对，特别是一些争议地区，故用的时候需强制更换底图)
national_map = folium.Map(location=[35.3, 100.6], zoom_start=4)
national_map.save(r'中国地图.html')

# 国内目前常见的坐标系主要分为三种：
# 地球坐标系——WGS84：常见于GPS设备，Google地图等国际标准的坐标体系。
# 火星坐标系——GCJ-02：中国国内使用的被强制加密后的坐标体系，高德坐标就属于该种坐标体系。
# 百度坐标系——BD-09：百度地图所使用的坐标体系，是在火星坐标系的基础上又进行了一次加密处理。
# folium库默认是基于OpenStreetMap的，但是可能由于信息更新不及时或者其他人为的原因，有时候OpenStreetMap的数据是不准确的。这就需要更换底图，如高德地图，或者Google地图等。
# 需要注意的是，更换了地图，对应的坐标系也需要跟着更换；
# tiles='OpenStreetMap'  时，对应的坐标系坐标是WGS84
# tiles='http://wprd04.is.autonavi.com/appmaptile?x={x}&y={y}&z={z}&lang=zh_cn&size=1&scl=1&style=7', # 高德街道图，新版；需要是GCJ02，高德坐标

m = folium.Map(
        location=[35.3, 100.6],
        zoom_start=4,
        # tiles='http://webrd02.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=7&x={x}&y={y}&z={z}', # 高德街道图
        # tiles='http://webrd02.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=7&x={x}&y={y}&z={z}', # 高德街道图,旧版
        # tiles='http://wprd04.is.autonavi.com/appmaptile?x={x}&y={y}&z={z}&lang=zh_cn&size=1&scl=1&style=7', # 高德街道图，新版；
        tiles='http://webst02.is.autonavi.com/appmaptile?style=6&x={x}&y={y}&z={z}', # 高德卫星图
        # tiles='https://mt.google.com/vt/lyrs=s&x={x}&y={y}&z={z}', # google 卫星图
        # tiles='https://mt.google.com/vt/lyrs=h&x={x}&y={y}&z={z}', # google 地图
        attr='default'
    )
m.save(r'D:\Users\abcd\Downloads\fpgrowth_code\中国地图.html')
# 如果需要更换底图，仅需调整tiles参数URL即可。
# 目前高德的瓦片地址有如下两种：
#
# http://wprd0{1-4}.is.autonavi.com/appmaptile?x={x}&y={y}&z={z}&lang=zh_cn&size=1&scl=1&style=7和
# http://webst0{1-4}.is.autonavi.com/appmaptile?style=7&x={x}&y={y}&z={z}
# 前者是高德的新版地址，后者是老版地址。
#
# 高德新版的参数：
#
# lang可以通过zh_cn设置中文，en设置英文；
# size基本无作用；
# scl设置标注还是底图，scl=1代表注记，scl=2代表底图（矢量或者影像）；
# style设置影像和路网，style=6为影像图，style=7为矢量路网，style=8为影像路网。
# 总结之：
#
# http://wprd0{1-4}.is.autonavi.com/appmaptile?x={x}&y={y}&z={z}&lang=zh_cn&size=1&scl=1&style=7 为矢量图（含路网、含注记）
# http://wprd0{1-4}.is.autonavi.com/appmaptile?x={x}&y={y}&z={z}&lang=zh_cn&size=1&scl=2&style=7 为矢量图（含路网，不含注记）
# http://wprd0{1-4}.is.autonavi.com/appmaptile?x={x}&y={y}&z={z}&lang=zh_cn&size=1&scl=1&style=6 为影像底图（不含路网，不含注记）
# http://wprd0{1-4}.is.autonavi.com/appmaptile?x={x}&y={y}&z={z}&lang=zh_cn&size=1&scl=2&style=6 为影像底图（不含路网、不含注记）
# http://wprd0{1-4}.is.autonavi.com/appmaptile?x={x}&y={y}&z={z}&lang=zh_cn&size=1&scl=1&style=8 为影像路图（含路网，含注记）
# http://wprd0{1-4}.is.autonavi.com/appmaptile?x={x}&y={y}&z={z}&lang=zh_cn&size=1&scl=2&style=8 为影像路网（含路网，不含注记）
# 高德旧版可以通过style参数设置影像、矢量、路网。
#
# 总结之：
#
# http://webst0{1-4}.is.autonavi.com/appmaptile?style=6&x={x}&y={y}&z={z} 为影像底图（不含路网，不含注记）
# http://webst0{1-4}.is.autonavi.com/appmaptile?style=7&x={x}&y={y}&z={z} 为矢量地图（含路网，含注记）
# http://webst0{1-4}.is.autonavi.com/appmaptile?style=8&x={x}&y={y}&z={z} 为影像路网（含路网，含注记）

# 城市地图
# 除了上述正常的地图显示外，folium 还提供了非常丰富的多样化显示，控制显示效果的变量是tiles，样式有"OpenStreetMap", "Stamen Terrain", "Stamen Toner", "Stamen Watercolor", "CartoDB positron", "CartoDB dark_matter"
# tiles='OpenStreetMap'  时，对应的坐标系坐标是WGS84
# tiles='http://wprd04.is.autonavi.com/appmaptile?x={x}&y={y}&z={z}&lang=zh_cn&size=1&scl=1&style=7', # 高德街道图，新版；需要是GCJ02，高德坐标

city_map = folium.Map(location=[22.68, 114.15], zoom_start=11, tiles='Stamen Terrain')  # zoom_start 数值越大，地图也会放大；
# display city map
city_map.save(r'D:\Users\abcd\Downloads\fpgrowth_code\深圳市.html')

# 在地图上标记
# 添加普通标记用 Marker
# 这里可以选择标记的图案。
# 中文转换
def parse_zhch(s):
    return str(str(s).encode('ascii' , 'xmlcharrefreplace'))[2:-1]

bj_map = folium.Map(location=[22.68, 114.15], zoom_start=11, tiles='Stamen Terrain')

folium.Marker(
    location=[22.643, 114.057],
    # popup=parse_zhch('位置1'),  # 若设置popup参数，则只能显示一个点；
    icon=folium.Icon(icon='cloud')
).add_to(bj_map)

folium.Marker(
    location=[22.497, 113.882],
    # popup=parse_zhch('位置2'),
    icon=folium.Icon(color='green')
).add_to(bj_map)

folium.Marker(
    location=[22.54, 113.951],
    # popup=parse_zhch('位置3'),
    icon=folium.Icon(color='red', icon='info-sign')
).add_to(bj_map)

bj_map.save(r'深圳市-位置标记.html')

# 添加圆形标记用 Circle 以及 CircleMarker
bj_map = folium.Map(location=[39.93, 116.40], zoom_start=12, tiles='Stamen Toner')

folium.Circle(
    radius=200,
    location=[39.92, 116.43],
    popup='The Waterfront',
    color='crimson',
    fill=False,
).add_to(bj_map)

folium.CircleMarker(
    location=[39.93, 116.38],
    radius=50,
    popup='Laurelhurst Park',
    color='#3186cc',
    fill=True,
    fill_color='#3186cc'
).add_to(bj_map)

bj_map.save('添加圆形标记.html')


# 点击获取经纬度,通过点击鼠标便可以获取点击出的经纬度。
m = folium.Map(location=[46.1991, -122.1889],tiles='Stamen Terrain',zoom_start=13)
m.add_child(folium.LatLngPopup())
m.save('点击获取经纬度.html')

# 动态放置标记
m = folium.Map(
    location=[46.8527, -121.7649],
    tiles='Stamen Terrain',
    zoom_start=13
)
folium.Marker(
    [46.8354, -121.7325],
    popup='Camp Muir'
).add_to(m)
m.add_child(folium.ClickForMarker(popup='Waypoint'))
m.save('动态放置标记.html')

# 热力图绘制
# 因为没有实际的经纬度坐标数据，所以这里只能模拟一些位置出来，另外每个位置还需要一个数值作为热力值。

# generated data
import numpy as np
data = (
    np.random.normal(size=(100, 3)) *
    np.array([[0.1, 0.1, 0.1]]) +
    np.array([[40, 116.5, 1]])
).tolist()
# data[:3]
# 数据分布
#
# [[40.04666663299843, 116.59569796477264, 0.9667425547098781],
#  [39.86836537517533, 116.28201445195315, 0.8708549157348728],
#  [40.08123232852134, 116.56884585184197, 0.9104952244371285]]

# 绘制热力图

# HeatMap
from folium.plugins import HeatMap
m = folium.Map([39.93, 116.38], tiles='stamentoner', zoom_start=6)
HeatMap(data).add_to(m)
# m.save(os.path.join('results', 'Heatmap.html'))
m.save('热力图绘制.html')


# 密度地图绘制
# folium 不仅可以绘制热力图，还可以绘制密度地图，按照经纬度进行举例聚类，然后在地图中显示。

from folium.plugins import MarkerCluster

m = folium.Map([39.93, 116.38], tiles='stamentoner', zoom_start=10)

# create a mark cluster object
marker_cluster = MarkerCluster().add_to(m)

# add data point to the mark cluster
for lat, lng, label in data:
    folium.Marker(
        location=[lat, lng],
        icon=None,
        popup=label,
    ).add_to(marker_cluster)

# add marker_cluster to map
m.add_child(marker_cluster)

def main():
    pass


if __name__ == '__main__':
    main()