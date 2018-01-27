#!/usr/bin/python3
# coding: utf-8

# bson是由10gen开发的一个数据格式，目前主要用于mongoDB中，是mongoDB的数据存储格式。bson基于json格式，选择json进行改造的原因主要是json的通用性及json的schemaless的特性。
# 数据结构：
# 　　json是像字符串一样存储的，bson是按结构存储的（像数组 或者说struct）

# 存储空间
# 　　bson>json

# 操作速度
# 　　bson>json。比如，遍历查找：json需要扫字符串，而bson可以直接定位

# 修改：
# 　　json也要大动大移，bson就不需要。

# gswyhq@gswyhq-PC:~$ sudo pip3 install bson

import bson

with open('/home/gswyhq/docker/mongo/data/mongodb_192_168_3_130/page/保险条款.bson', 'rb')as f:
    b_data = f.read()

data = bson.loads(b_data)
type(data)
# Out[5]: dict
list(data.keys())[:3]
# Out[6]: ['_id', 'content', '保险小类']
type(data.get('保险小类'))
# Out[7]: str


def main():
    pass


if __name__ == '__main__':
    main()