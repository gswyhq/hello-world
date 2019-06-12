
1 哨兵的作用
哨兵是redis集群架构中非常重要的一个组件，主要功能如下： 

集群监控：负责监控redis master和slave进程是否正常工作 
消息通知：如果某个redis实例有故障，那么哨兵负责发送消息作为报警通知给管理员 
故障转移：如果master node挂掉了，会自动转移到slave node上 
配置中心：如果故障转移发生了，通知client客户端新的master地址
2 哨兵的核心知识
故障转移时，判断一个master node是宕机了，需要大部分的哨兵都同意才行，涉及到了分布式选举的问题
哨兵至少需要3个实例，来保证自己的健壮性
哨兵 + redis主从的部署架构，是不会保证数据零丢失的，只能保证redis集群的高可用性

# 准备
创建redis-master/slave容器、封装redis-sentinel镜像，同单机部署

docker pull redis:3.2.12

http://download.redis.io/releases/redis-3.2.12.tar.gz

```shell
redis-master： 192.168.3.164:7687
redis-slave1： 192.168.3.132:7687
redis-slave2： 192.168.3.133:7687
redis-sentinel1: 192.168.3.164:26379
redis-sentinel2: 192.168.3.132:26379
redis-sentinel3: 192.168.3.133:26379
```

# 部署

```shell
root@192.168.3.164:~$ docker-compose -f docker-compose-164.yml up -d
root@192.168.3.132:~$ docker-compose -f docker-compose-132-133.yml up -d
root@192.168.3.133:~$ docker-compose -f docker-compose-132-133.yml up -d

# redis-sentinel集群外网访问
# 使用docker部署redis sentinel集群时，若要对宿主机外部提供服务，需要在配置各个容器时使用host网络模式；
# 默认的bridge模式将导致sentinel容器返回master容器的内部IP，外部无法访问。

```

# 使用
```python
# 连接示例
import redis
from redis.sentinel import Sentinel
REDIS_SENTINEL_SENTINELS = [('192.168.3.164', 26379),
                            ('192.168.3.132', 26379),
                            ('192.168.3.133', 26379)]
REDIS_SENTINEL_PASSWORD = 'redisweb1123'

sentinel = Sentinel(REDIS_SENTINEL_SENTINELS, 
                    socket_timeout=0.5)

# 主节点读写
master = sentinel.master_for('mymaster', socket_timeout=0.5, password=REDIS_SENTINEL_PASSWORD, db=0)
master.get('foo')
master.set('foo', 123)
Out[115]: True
master.get('foo')
Out[116]: b'123'

# 从节点读
# 获取从服务器进行读取（默认是round-roubin）
slave = sentinel.slave_for('mymaster', socket_timeout=0.5, password=REDIS_SENTINEL_PASSWORD, db=0)
slave.get('foo')
Out[118]: b'123'
```

[参考阿里云部署高可用Redis集群](https://yq.aliyun.com/articles/57953)

https://github.com/AliyunContainerService/redis-cluster.git


