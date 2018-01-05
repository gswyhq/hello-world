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

简单的redis操作
redis连接实例是线程安全的，可以直接将redis连接实例设置为一个全局变量，直接使用。如果需要另一个
Redis实例（or Redis数据库）时，就需要重新创建redis连接实例来获取一个新的连接。同理，python的redis没有实现select命令。

>>> import redis
>>> r = redis.Redis(host='localhost',port=6379,db=0)
>>> r.set('guo','shuai')
True
>>> r.get('guo')
'shuai'
>>> r['guo']            
'shuai'
>>> r.keys()
['guo']
>>> r.dbsize()         #当前数据库包含多少条数据       
1L
>>> r.delete('guo')
1
>>> r.save()               #执行“检查点”操作，将数据写回磁盘。保存时阻塞
True
>>> r.get('guo');
>>> r.flushdb()        #清空r中的所有数据
True
3. pipeline操作
管道（pipeline）是redis在提供单个请求中缓冲多条服务器命令的基类的子类。它通过减少服务器-客户端之间反复的TCP数据库包，从而大大提高了执行批量命令的功能。


>>> p = r.pipeline()        --创建一个管道
>>> p.set('hello','redis')
>>> p.sadd('faz','baz')
>>> p.incr('num')
>>> p.execute()
[True, 1, 1]
>>> r.get('hello')
'redis'


>>> p = r.pipeline()        --创建一个管道
>>> p.set('hello','redis')
>>> p.sadd('faz','baz')
>>> p.incr('num')
# 一次性执行上边的三个命令
>>> p.execute()
[True, 1, 1]
>>> r.get('hello')
'redis'
管道的命令可以写在一起，如：
>>> p.set('hello','redis').sadd('faz','baz').incr('num').execute()
1
>>> p.set('hello','redis').sadd('faz','baz').incr('num').execute()
默认的情况下，管道里执行的命令可以保证执行的原子性，执行pipe = r.pipeline(transaction=False)可以禁用这一特性。

4. 应用场景 – 页面点击数
《Redis Cookbook》对这个经典场景进行详细描述。假定我们对一系列页面需要记录点击次数。例如论坛的每个帖子都要记录点击次数，而点击次数比回帖的次数的多得多。如果使用关系数据库来存储点击，可能存在大量的行级锁争用。所以，点击数的增加使用redis的INCR命令最好不过了。
当redis服务器启动时，可以从关系数据库读入点击数的初始值（1237这个页面被访问了34634次）


>>> r.set("visit:1237:totals",34634)
True
1
2
>>> r.set("visit:1237:totals",34634)
True
每当有一个页面点击，则使用INCR增加点击数即可。


>>> r.incr("visit:1237:totals")
34635
>>> r.incr("visit:1237:totals")
34636


>>> r.incr("visit:1237:totals")
34635
>>> r.incr("visit:1237:totals")
34636
页面载入的时候则可直接获取这个值


>>> r.get ("visit:1237:totals")
'34636'
1
2
>>> r.get ("visit:1237:totals")
'34636'
5. 使用hash类型保存多样化对象
当有大量类型文档的对象，文档的内容都不一样时，（即“表”没有固定的列），可以使用hash来表达。


>>> r.hset('users:jdoe',  'name', "John Doe")
1L
>>> r.hset('users:jdoe', 'email', 'John@test.com')
1L
>>> r.hset('users:jdoe',  'phone', '1555313940')
1L
>>> r.hincrby('users:jdoe', 'visits', 1)
1L
>>> r.hgetall('users:jdoe')
{'phone': '1555313940', 'name': 'John Doe', 'visits': '1', 'email': 'John@test.com'}
>>> r.hkeys('users:jdoe')
['name', 'email', 'phone', 'visits']


>>> r.hset('users:jdoe',  'name', "John Doe")
1L
>>> r.hset('users:jdoe', 'email', 'John@test.com')
1L
>>> r.hset('users:jdoe',  'phone', '1555313940')
1L
>>> r.hincrby('users:jdoe', 'visits', 1)
1L
>>> r.hgetall('users:jdoe')
{'phone': '1555313940', 'name': 'John Doe', 'visits': '1', 'email': 'John@test.com'}
>>> r.hkeys('users:jdoe')
['name', 'email', 'phone', 'visits']
6. 应用场景 – 社交圈子数据
在社交网站中，每一个圈子(circle)都有自己的用户群。通过圈子可以找到有共同特征（比如某一体育活动、游戏、电影等爱好者）的人。当一个用户加入一个或几个圈子后，系统可以向这个用户推荐圈子中的人。
我们定义这样两个圈子,并加入一些圈子成员。


>>> r.sadd('circle:game:lol','user:debugo')
1
>>> r.sadd('circle:game:lol','user:leo')
1
>>> r.sadd('circle:game:lol','user:Guo')
1
>>> r.sadd('circle:soccer:InterMilan','user:Guo')
1
>>> r.sadd('circle:soccer:InterMilan','user:Levis')
1
>>> r.sadd('circle:soccer:InterMilan','user:leo')
1


>>> r.sadd('circle:game:lol','user:debugo')
1
>>> r.sadd('circle:game:lol','user:leo')
1
>>> r.sadd('circle:game:lol','user:Guo')
1
>>> r.sadd('circle:soccer:InterMilan','user:Guo')
1
>>> r.sadd('circle:soccer:InterMilan','user:Levis')
1
>>> r.sadd('circle:soccer:InterMilan','user:leo')
1
#获得某一圈子的成员


>>> r.smembers('circle:game:lol')
set(['user:Guo', 'user:debugo', 'user:leo'])
redis> smembers circle:jdoe:family    
1
2
3
>>> r.smembers('circle:game:lol')
set(['user:Guo', 'user:debugo', 'user:leo'])
redis> smembers circle:jdoe:family    
可以使用集合运算来得到几个圈子的共同成员：


>>> r.sinter('circle:game:lol', 'circle:soccer:InterMilan')
set(['user:Guo', 'user:leo'])
>>> r.sunion('circle:game:lol', 'circle:soccer:InterMilan')
set(['user:Levis', 'user:Guo', 'user:debugo', 'user:leo'])
1
2
3
4
>>> r.sinter('circle:game:lol', 'circle:soccer:InterMilan')
set(['user:Guo', 'user:leo'])
>>> r.sunion('circle:game:lol', 'circle:soccer:InterMilan')
set(['user:Levis', 'user:Guo', 'user:debugo', 'user:leo'])
7. 应用场景 – 实时用户统计
Counting Online Users with Redis介绍了这个方法。当我们需要在页面上显示当前的在线用户时，就可以使用Redis来完成了。首先获得当前时间（以Unix timestamps方式）除以60，可以基于这个值创建一个key。然后添加用户到这个集合中。当超过你设定的最大的超时时间，则将这个集合设为过期；而当需要查询当前在线用户的时候，则将最后N分钟的集合交集在一起即可。由于redis连接对象是线程安全的，所以可以直接使用一个全局变量来表示。


import time
from redis import Redis
from datetime import datetime
ONLINE_LAST_MINUTES = 5
redis = Redis()

def mark_online(user_id):         #将一个用户标记为online
    now = int(time.time())        #当前的UNIX时间戳
    expires = now + (app.config['ONLINE_LAST_MINUTES'] * 60) + 10    #过期的UNIX时间戳
    all_users_key = 'online-users/%d' % (now // 60)        #集合名，包含分钟信息
    user_key = 'user-activity/%s' % user_id                
    p = redis.pipeline()
    p.sadd(all_users_key, user_id)                         #将用户id插入到包含分钟信息的集合中
    p.set(user_key, now)                                   #记录用户的标记时间
    p.expireat(all_users_key, expires)                     #设定集合的过期时间为UNIX的时间戳
    p.expireat(user_key, expires)
    p.execute()

def get_user_last_activity(user_id):        #获得用户的最后活跃时间
    last_active = redis.get('user-activity/%s' % user_id)  #如果获取不到，则返回None
    if last_active is None:
        return None
    return datetime.utcfromtimestamp(int(last_active))

def get_online_users():                     #获得当前online用户的列表
    current = int(time.time()) // 60        
    minutes = xrange(app.config['ONLINE_LAST_MINUTES'])
    return redis.sunion(['online-users/%d' % (current - x)        #取ONLINE_LAST_MINUTES分钟对应集合的交集
                         for x in minutes])


import time
from redis import Redis
from datetime import datetime
ONLINE_LAST_MINUTES = 5
redis = Redis()
 
def mark_online(user_id):         #将一个用户标记为online
    now = int(time.time())        #当前的UNIX时间戳
    expires = now + (app.config['ONLINE_LAST_MINUTES'] * 60) + 10    #过期的UNIX时间戳
    all_users_key = 'online-users/%d' % (now // 60)        #集合名，包含分钟信息
    user_key = 'user-activity/%s' % user_id                
    p = redis.pipeline()
    p.sadd(all_users_key, user_id)                         #将用户id插入到包含分钟信息的集合中
    p.set(user_key, now)                                   #记录用户的标记时间
    p.expireat(all_users_key, expires)                     #设定集合的过期时间为UNIX的时间戳
    p.expireat(user_key, expires)
    p.execute()
 
def get_user_last_activity(user_id):        #获得用户的最后活跃时间
    last_active = redis.get('user-activity/%s' % user_id)  #如果获取不到，则返回None
    if last_active is None:
        return None
    return datetime.utcfromtimestamp(int(last_active))
 
def get_online_users():                     #获得当前online用户的列表
    current = int(time.time()) // 60        
    minutes = xrange(app.config['ONLINE_LAST_MINUTES'])
    return redis.sunion(['online-users/%d' % (current - x)        #取ONLINE_LAST_MINUTES分钟对应集合的交集
                         for x in minutes])


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

# decode_responses=True: 防止get到的值为`bytes`
# 默认情况下，publish的消息会被编码，当你获取消息时得到的是编码后的字节，如果你需要它自动解码，创建Redis client实例时需要指定decode_responses=True,
# (译者注：不建议使用该选项，因为当存在pickle序列化的值时，client.get(key)时会出现解码失败的错误UnicodeDecodeError)


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

#!/usr/bin/python
import redis
import time
## Connect local redis service
client =redis.Redis(host='127.0.0.1',port=6379,db=0)
print "Connection to server successfully!"
dicKeys = client.keys("*")
print dicKeys

### Redis hash command part Start ###
# hset: Set key to value with hash name,hset(self, name, key, value)
# hget: Return the value of ``key`` within the hash ``name``, hget(self, name, key)
client.hset('myhash','field1',"foo")
hashVal = client.hget('myhash','field1')
print "Get hash value:",hashVal

# Get none-value
hashVal = client.hget('myhash','field2')
print "None hash value:",hashVal

# hexists: Returns a boolean indicating if ``key`` exists within hash ``name``
keyList= ['field1','field2']
for key in keyList:
    hexists = client.hexists('myhash',key)
    if hexists :
        print "Exist in redis-hash key:",key
    else:
        print "Not exist in redis-hash key:",key

# hgetall: Return a Python dict of the hash's name/value pairs
client.hset('myhash','field2',"bar")
valDict = client.hgetall('myhash')
print "Get python-dict from redis-hash",valDict

# hincrby: Increment the value of ``key`` in hash ``name`` by ``amount``
# default increment is 1,
client.hset('myhash','field',20)
client.hincrby('myhash','field')
print "Get incrby value(Default):",client.hget('myhash','field')
client.hincrby('myhash','field',2)
print "Get incrby value(step: 2):",client.hget('myhash','field')
client.hincrby('myhash','field',-3)
print "Get incrby value(step: -3):",client.hget('myhash','field')

# no method hincrbyfloat

#hkeys: Return the list of keys within hash ``name``
kL = client.hkeys('myhash')
print "Get redis-hash key list",kL

#hlen: Return the number of elements in hash ``name``
lenHash =client.hlen('myhash')
print "All hash length:",lenHash

#hmget: Returns a list of values ordered identically to ``keys``
#hmget(self, name, keys), keys should be python list data structure
val =client.hmget('myhash',['field','field1','field2','field3','fieldx'])
print "Get all redis-hash value list:",val

#hmset:  Sets each key in the ``mapping`` dict to its corresponding value in the hash ``name``
hmDict={'field':'foo','field1':'bar'}
hmKeys=hmDict.keys()
client.hmset('hash',hmDict)
val = client.hmget('hash',hmKeys)
print "Get hmset value:",val

#hdel: Delete ``key`` from hash ``name``
client.hdel('hash','field')
print "Get delete result:",client.hget('hash','field')

#hvals:  Return the list of values within hash ``name``
val = client.hvals('myhash')
print "Get redis-hash values with HVALS",val

#hsetnx: Set ``key`` to ``value`` within hash ``name`` if ``key`` does not exist.
#      Returns 1 if HSETNX created a field, otherwise 0.
r=client.hsetnx('myhash','field',2)
print "Check hsetnx execute result:",r," Value:",client.hget('myhash','field')
r=client.hsetnx('myhash','field10',20)
print "Check hsetnx execute result:",r,"Value",client.hget('myhash','field10')

hashVal = client.hgetall('profile')
print hashVal
#Empty db
client.flushdb()


