#!/usr/bin/python
# -*- coding:utf8 -*- #
from __future__ import  generators
from __future__ import  division
from __future__ import  print_function
from __future__ import  unicode_literals
import sys,os,json

if sys.version >= '3':
    PY3 = True
else:
    PY3 = False

if PY3:
    import pickle
else:
    import cPickle as pickle
    from codecs import open

import time, json
from pykafka import KafkaClient
 
# 相关的mysql操作
mysql_op()
 
# 可接受多个Client这是重点
client = KafkaClient(hosts="192.168.1.233:9092, \
                            192.168.1.233:9093, \
                            192.168.1.233:9094")
# 选择一个topic
topic = client.topics['goods-topic']
# 创建一个生产者
producer = topic.get_producer()
# 模拟接收前端生成的商品信息
goods_dict = {
  'option_type':'insert',
  'option_obj':{
    'goods_name':'goods-1',
    'goods_price':10.00,
    'create_time':time.strftime('%Y-%m-%d %H:%M:%S')
  }
}
goods_json = json.dumps(goods_dict)
# 生产消息
producer.produce(msg)

def main():
    pass


if __name__ == "__main__":
    main()
