
Redis-Sentinel是Redis官方推荐的高可用性(HA)解决方案，当用Redis做Master-slave的高可用方案时，
假如master宕机了，Redis本身(包括它的很多客户端)都没有实现自动进行主备切换，
而Redis-sentinel本身也是一个独立运行的进程，它能监控多个master-slave集群，
发现master宕机后能进行自动切换。

# 创建redis-master/slave容器

docker pull redis:3.2.12

http://download.redis.io/releases/redis-3.2.12.tar.gz

```shell
本机地址： 192.168.3.164

redis-master地址： 192.168.3.164:7687
redis-slave1地址： 192.168.3.164:7688
redis-slave2地址： 192.168.3.164:7689
redis-slave3地址： 192.168.3.164:7690
```

在redis源代码中提供的redis.conf的基础上进行修改
`bind 127.0.0.1` 改为 `bind 0.0.0.0`

masterauth <master-password> 当master服务设置了密码保护时，slave服务连接master的密码
requirepass<password> 设置Redis连接密码

redis-master:
`docker run --rm -v $PWD/redis.conf:/redis.conf -net=host redis:3.2.12 redis-server /redis.conf --port 7687 --requirepass "redisweb1123" --masterauth "redisweb1123"`

redis-slave1:
`docker run --rm -v $PWD/redis.conf:/redis.conf -net=host redis:3.2.12 redis-server /redis.conf --port 7688 --slaveof 192.168.3.164 7687 --requirepass "redisweb1123" --masterauth "redisweb1123"`

redis-slave2:
`docker run --rm -v $PWD/redis.conf:/redis.conf -net=host redis:3.2.12 redis-server /redis.conf --port 7689 --slaveof 192.168.3.164 7687 --requirepass "redisweb1123" --masterauth "redisweb1123"`

redis-slave3:
`docker run --rm -v $PWD/redis.conf:/redis.conf -net=host redis:3.2.12 redis-server /redis.conf --port 7690 --slaveof 192.168.3.164 7687 --requirepass "redisweb1123" --masterauth "redisweb1123"`

# 封装redis-sentinel镜像
sentinel$ ls
Dockerfile  sentinel.conf  sentinel-entrypoint.sh

`sentinel.conf` 文件中设置对应的`redis-master`地址
`sentinel monitor mymaster 192.168.3.164 7687 $SENTINEL_QUORUM`

构建镜像
`~sentinel$: docker build -t redis-sentinel:26379 --no-cache -f Dockerfile .`

# 通过docker-compose-single.yml部署集群及哨兵服务
```shell
redis-sentinel1: 192.168.3.164:26379
redis-sentinel2: 192.168.3.164:26380
redis-sentinel3: 192.168.3.164:26381
```

`docker-compose -f docker-compose-single.yml up -d `

# 使用
```python
# 连接示例
import redis
from redis.sentinel import Sentinel
REDIS_SENTINEL_SENTINELS = [('192.168.3.164', 26379),
                            ('192.168.3.164', 26380),
                            ('192.168.3.164', 26381)]
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




