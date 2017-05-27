#!/usr/bin/python
# -*- coding:utf8 -*- #
from __future__ import generators
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import sys

if sys.version >= '3':
    PY3 = True
else:
    PY3 = False

if PY3:
    import pickle
else:
    import cPickle as pickle
    from codecs import open

import redis
r = redis.StrictRedis(host='localhost', port=6379, db=0)
r.set('foo', 'bar')#添加一个键值对

r.get('foo') #获取键的值

r.hset('users:jdoe',  'name', "John Doe") #设置字段（列名），键值对

r.hset('users:jdoe', 'email', 'John@test.com')

r.hset('users:jdoe',  'phone', '1555313940')

r.hincrby('users:jdoe', 'visits', 1)

r.hgetall('users:jdoe') #获取'users:jdoe'字段的所有键值对，得到的是一个dict
r.hkeys('users:jdoe') ##获取'users:jdoe'字段的键，得到的是一个list

p = r.pipeline()      #  --创建一个管道
p.set('hello','redis')
p.sadd('faz','baz')
p.incr('num')
p.execute()

r.get('hello')

#管道的命令可以写在一起，如：
#p.set('hello','redis').sadd('faz','baz').incr('num').execute()
#默认的情况下，管道里执行的命令可以保证执行的原子性，执行pipe = r.pipeline(transaction=False)可以禁用这一特性。

r.set('visit',10) #存储在redis中的键值对，即使是int类型等，会自动转化成str。
r.incr('visit') #使用INCR,可以使对应键值加1

#定义两个圈子（键），添加成员
r.sadd('circle:game:lol','user:debugo')
r.sadd('circle:game:lol','user:leo')
r.sadd('circle:game:lol','user:Guo')
r.sadd('circle:soccer:InterMilan','user:Guo')
r.sadd('circle:soccer:InterMilan','user:Levis')
r.sadd('circle:soccer:InterMilan','user:leo')

#获取圈子的值，实际上获取的是一个set
r.smembers('circle:game:lol')

def main():
    pass


if __name__ == "__main__":
    main()
   

redis 聚群：
sudo pip2 install redis-py-cluster

>>> from rediscluster import RedisCluster
>>> startup_nodes = [{'host': '172.19.1.106', 'port': '7001'},
                         {'host': '172.19.1.106', 'port': '7002'},
                         {'host': '172.19.1.106', 'port': '7003'},
                         {'host': '172.19.1.106', 'port': '7004'},
                         {'host': '172.19.1.106', 'port': '7005'},
                         {'host': '172.19.1.106', 'port': '7006'}]
>>> rc = RedisCluster(startup_nodes=startup_nodes, decode_responses=True)
>>> rc.set("foo", "bar")
True
>>> rc.get("foo")
'bar'

 
    HDEL: 删除对应哈希（Hash）表的指定键（key）的字段，hdel(self, name, key)
HEXISTS: 检测哈希（Hash）表对应键（key）字段是否存在，返回布尔逻辑，hexists(self, name, key)
HGET: 获取哈希（Hash）指定键（key）对应的值，hget(self, name, key)
HGETALL: 获取哈希(Hash)表的键-值对（key-value pairs）,返回python字典类型数据，hgetall(self, name)
HINCRBY: 为哈希表（Hash）指定键（key）对应的值（key）加上指定的整数数值（int，可为负值），参见 [Python操作Redis：字符串(String)]，(http://blog.csdn.net/u012894975/article/details/51285733)hincrby(self, name, key, amount=1)，Redis 中本操作的值被限制在 64 位(bit)有符号数字。
HKEYS: 返回哈希表（Hash）对应键（key）的数组（Python称之为列表List），hkeys(self, name)
HLEN: 获取哈希表（Hash）中键-值对（key-value pairs）个数，hlen(self, name)
HMGET: 获取哈希表（Hash）中一个或多个给点字段的值，不存在返回nil(Redis命令行)/None(Python)，hmget(self, name, keys)，其中keys可以为列表（list）
HMSET: 设置对个键-值对（key-value pairs）到哈希表（Hash）中，python输入值（mapping）为字典（dictionary）类型，hmset(self, name, mapping)
HSET: 为哈希表（Hash）赋值，若键（key）存在值（value）则覆盖，不存在则创建，hset(self, name, key, value)
HSETNX：为哈希表（Hash）不存值（value）的键（key）赋值，存在操作无效，对应值（value）无变化，hsetnx(self, name, key, value)
HVALS：返回哈希表（Hash）对应值（value）的列表，hvals(self, name)

