#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 根据经纬度点求面积
# 根据经纬度计算的面积，需要一个转化，才能和高德地图显示的面积是一致的。

# 这里有个前提是：你的经纬度点是依次相连接的点，不是无序的，可以是顺时针，或者逆时针都可以。

import math
def ComputeArea(data):
    arr = data.split(';')
    arr_len = len(arr)
    if arr_len < 3:
        return 0.0
    temp = []
    for i in range(0,arr_len):
        temp.append([float(x) for x in arr[i].split(',')])
    s = temp[0][1] * (temp[arr_len -1][0]-temp[1][0])
    print (s)
    for i in range(1,arr_len):
        s += temp[i][1] * (temp[i-1][0] - temp[(i+1)%arr_len][0])
    return round(math.fabs(s/2)*9101160000.085981,6)

# 最后求的面积是：
# 38602.365364

def main():
    data = "115.989099,39.646023;115.987394,39.645988;115.987371,39.647407;115.986684,39.647423;115.986602,39.648088;115.989095,39.648151;115.989188,39.646021;115.989099,39.646023"
    ComputeArea(data)

if __name__ == '__main__':
    main()