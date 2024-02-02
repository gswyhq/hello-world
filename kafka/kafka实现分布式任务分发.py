#!/usr/bin/env python
# coding=utf-8

# pip3 install kafka-python==2.0.2
# 创建主题：
# [root@d35e11a780bd kafka]# bin/kafka-topics.sh  --zookeeper 192.168.3.105:2181 --create --topic test3  --partitions 1 --replication-factor 1
# Created topic "test3".

###############################################################################################################################
# 显示所有主题
像这样:
./bin/kafka-topics.sh --list --zookeeper localhost:2181

import kafka
consumer = kafka.KafkaConsumer(group_id='test', bootstrap_servers=['192.168.3.105:9092'])
consumer.topics()

###############################################################################################################################
# kafka中group-id作用
topic到group之间是发布订阅的通信方式，即一条topic会被所有的group消费，属于一对多模式；group到consumer是点对点通信方式，属于一对一模式。
举例:
不使用group的话，启动10个consumer消费一个topic，这10个consumer都能得到topic的所有数据，相当于这个topic中的任一条消息被消费10次。
使用group的话，连接时带上groupid，topic的消息会分发到10个consumer上，每条消息只被消费1次。

###############################################################################################################################
# 生产者：
>>> from kafka import KafkaProducer
>>> producer = KafkaProducer(bootstrap_servers='localhost:1234')
>>> for _ in range(100):
...     producer.send('foobar', b'some_message_bytes')
>>> # Block until a single message is sent (or timeout)
>>> future = producer.send('foobar', b'another_message')
>>> result = future.get(timeout=60)
>>> # Block until all pending messages are at least put on the network
>>> # NOTE: This does not guarantee delivery or success! It is really
>>> # only useful if you configure internal batching using linger_ms
>>> producer.flush()
>>> # Use a key for hashed-partitioning
>>> producer.send('foobar', key=b'foo', value=b'bar')
>>> # Serialize json messages
>>> import json
>>> producer = KafkaProducer(value_serializer=lambda v: json.dumps(v).encode('utf-8'))
>>> producer.send('fizzbuzz', {'foo': 'bar'})
>>> # Serialize string keys
>>> producer = KafkaProducer(key_serializer=str.encode)
>>> producer.send('flipflap', key='ping', value=b'1234')
>>> # Compress messages
>>> producer = KafkaProducer(compression_type='gzip')
>>> for i in range(1000):
...     producer.send('foobar', b'msg %d' % i)

###############################################################################################################################
# 消费者：
>>> from kafka import KafkaConsumer
>>> consumer = KafkaConsumer('my_favorite_topic')
>>> for msg in consumer:
...     print (msg)
>>> # join a consumer group for dynamic partition assignment and offset commits
>>> from kafka import KafkaConsumer
>>> consumer = KafkaConsumer('my_favorite_topic', group_id='my_favorite_group')
>>> for msg in consumer:
...     print (msg)
>>> # manually assign the partition list for the consumer
>>> from kafka import TopicPartition
>>> consumer = KafkaConsumer(bootstrap_servers='localhost:1234')
>>> consumer.assign([TopicPartition('foobar', 2)])
>>> msg = next(consumer)
>>> # Deserialize msgpack-encoded values
>>> consumer = KafkaConsumer(value_deserializer=msgpack.loads)
>>> consumer.subscribe(['msgpackfoo'])
>>> for msg in consumer:
...     assert isinstance(msg.value, dict)

# 资料来源：
# https://kafka-python.readthedocs.io/en/master/

def main():
    pass


if __name__ == "__main__":
    main()
