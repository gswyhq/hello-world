#!/usr/bin/env python
# coding=utf-8

import time
import redis # pip install redis==4.5.5
from redis.exceptions import RedisClusterException

redis_host = "192.168.3.105"
redis_port = 6379
redis_db = 0
redis_username = ''
redis_password = ''

try:
    redis_client = redis.RedisCluster(host=redis_host, port=redis_port, username=redis_username, password=redis_password)
    redis_client.info()
except RedisClusterException as e:
    redis_client = redis.Redis(host=redis_host, port=redis_port, db=redis_db, username=redis_username, password=redis_password)
    redis_client.info()

# 在分布式系统中，任务分发是一个非常重要的部分。任务分发可以保证任务在不同节点上的均衡分配，可以避免某个节点负载过高而导致的整个系统崩溃的风险。
# 在本文中，我们将介绍Redis的分布式任务分发方法并结合应用实例进行演示。本文的目的是帮助读者更好地理解和应用Redis在分布式系统中的优越性。
#
# Redis的分布式任务分发方法
# Redis是一个高效的NoSQL数据库，常用作缓存和数据存储。而在分布式系统中，Redis还可以作为任务分发的中心控制器，实现分布式任务分发的功能。

# 这种方式存在两个局限性：
# 不能支持一对多的消息分发。
# 如果生产者生成的速度远远大于消费者消费的速度，易堆积大量未消费的消息

# 在Redis中，我们可以利用它提供的pub/sub(发布/订阅)机制来实现任务分发。具体实现方法如下：

######################################################################################################################################
# 在该代码中，我们通过Redis的LPUSH命令将5个任务添加到队列中，并通过PUBLISH命令向频道发布任务信息。
import redis

task_queue = ['task1', 'task2', 'task3', 'task4', 'task5']
for task in task_queue:
    redis_client.lpush('task_queue', task) # 我们可以通过Redis的LPUSH命令将新的任务添加到队列中
    redis_client.publish('task_channel', f'{task} is added to task_queue.') # 通过Redis的PUBLISH命令发布该任务的信息，并通过频道来向其他节点广播此信息。

# pubsub channels [argument  [atgument ...] ]
# 查看订阅与发布系统的状态
# 说明：返回活跃频道列表（即至少有一个订阅者的频道，订阅模式的客户端除外）

######################################################################################################################################
# 接着，我们使用Python语言来实现订阅任务的代码（订阅者）：
import redis
import time

# 查看订阅与发布系统状态，因为还未开始订阅，这时频道列表是空的
task_channel = redis_client.pubsub()
# task_channel.channels
# Out[22]: {}

task_channel.subscribe('task_channel')  # 只能接收到订阅之后发布的消息，订阅之前发布的消息是接收不到的；
# 通过Redis的SUBSCRIBE命令订阅该频道信息。这样一来，一旦有新任务发布到频道中，订阅者就可以及时获取并开始执行任务。
# 通过Redis的SUBSCRIBE命令订阅了频道，并通过监听频道信息的方式来获取最新的任务。每当订阅者获取到新任务时，就会开始执行该任务。

# 因为订阅了，这时频道列表有值：
# task_channel.channels
# Out[24]: {b'task_channel': None}

while True:
    task = task_channel.get_message(ignore_subscribe_messages=True)
    if task and task['type'] == 'message':
        task_info = task['data']
        print(f'Received new task: {task_info}')
        task_name = str(task_info).split("'")[1]
        print(f'Starting to process task: {task_name}')
        time.sleep(2)
    
redis_client.unsubscribe('task_channel')
# 退订给定的频道; 若没有指定channel，则默认退订所有频道

# 注意：如果是先发布消息，再订阅频道，不会收到订阅之前就发布到该频道的消息！
# 使用注意
# 客户端需要及时消费和处理消息。
# 客户端订阅了channel之后，如果接收消息不及时，可能导致DCS实例消息堆积，当达到消息堆积阈值（默认值为32MB），或者达到某种程度（默认8MB）一段时间（默认为1分钟）后，服务器端会自动断开该客户端连接，避免导致内部内存耗尽。
# 客户端需要支持重连。
# 当连接断开之后，客户端需要使用subscribe或者psubscribe重新进行订阅，否则无法继续接收消息。
# 不建议用于消息可靠性要求高的场景中。
# Redis的pubsub不是一种可靠的消息系统。当出现客户端连接退出，或者极端情况下服务端发生主备切换时，未消费的消息会被丢弃。

######################################################################################################################################
# 当然，也可以异步订阅发布消息： https://redis.readthedocs.io/en/stable/examples/asyncio_examples.html
import asyncio

import redis.asyncio as async_redis

STOPWORD = "STOP"


async def reader(channel: async_redis.client.PubSub):
    while True:
        message = await channel.get_message(ignore_subscribe_messages=True)
        if message is not None:
            print(f"(Reader) Message Received: {message}")
            if message["data"].decode() == STOPWORD:
                print("(Reader) STOP")
                break
    
r = async_redis.from_url(f"redis://{redis_host}:{redis_port}")
async with r.pubsub() as pubsub:
    await pubsub.subscribe("channel:1", "channel:2")

    future = asyncio.create_task(reader(pubsub))

    await r.publish("channel:1", "Hello")
    await r.publish("channel:2", "World")
    await r.publish("channel:1", STOPWORD)

    await future
    
######################################################################################################################################

def main():
    pass


if __name__ == "__main__":
    main()
