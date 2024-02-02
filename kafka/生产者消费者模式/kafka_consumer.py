#!/usr/bin/env python
# coding=utf-8

import time
from datetime import datetime
import json
import kafka
from kafka import KafkaConsumer, TopicPartition, KafkaClient
from kafka.admin import KafkaAdminClient, NewTopic


bootstrap_servers = ['192.168.3.105:9092']
topic = 'test3'
group_id = 't1234'

NUM_PARTITIONS = 50

class Consumer():
    def __init__(self, topic=topic, group_id=group_id, bootstrap_servers=bootstrap_servers, api_version=(1, 0, 0)):
        self.consumer = kafka.KafkaConsumer(topic, group_id=group_id, bootstrap_servers=bootstrap_servers,
                                            api_version=api_version,
                                            auto_offset_reset='earliest',
                                            enable_auto_commit=True,
                                            value_deserializer=lambda m: json.loads(m.decode('utf-8'))
                                            )

        topics = self.consumer.topics()
        print("当前所有主题列表：{}".format(topics))
        if topic not in topics:  # Topic不存在
            admin = KafkaAdminClient(bootstrap_servers=bootstrap_servers, api_version=api_version)
            # 创建一个主题
            new_topic = NewTopic(topic, num_partitions=NUM_PARTITIONS, replication_factor=1)
            admin.create_topics([new_topic])

        self.group_id = group_id
        self.bootstrap_servers = bootstrap_servers
        self.api_version = api_version
        self.topic = topic

    def recv(self):
        """
         接收消费中的数据
         Returns:
           使用生成器进行返回;
         """
        for message in self.consumer:
            # 这是一个永久阻塞的过程,生产者消息会缓存在消息队列中,并且不删除,所以每个消息在消息队列中都会有偏移
            # print("主题:%s 分区:%d:连续值:%d: 键:key=%s 值:value=%s" % (
            #   message.topic, message.partition, message.offset, message.key, message.value))
            yield {"topic": message.topic, "partition": message.partition, "key": message.key,
              "value": message.value.decode('utf-8')}

    def test_poll(self, topic=None):
        if topic is None:
            topic = self.topic
        self.consumer.close()
        index = 0
        while True:
            consumer = kafka.KafkaConsumer(topic, group_id=self.group_id, bootstrap_servers=self.bootstrap_servers,
                                           api_version=self.api_version,
                                           auto_offset_reset='earliest',
                                           enable_auto_commit=True,
                                           value_deserializer=lambda m: json.loads(m.decode('utf-8'))
                                           )
            msg = consumer.poll(timeout_ms=5, max_records=1)  # 从kafka获取max_records条消息
            while not msg:
                msg = consumer.poll(timeout_ms=5, max_records=1)  # 从kafka获取max_records条消息
            # time.sleep(10)
            index += 1
            if msg:
                print(msg)
                for tp, records in msg.items():
                    for record in records:
                        print(record.offset, record.value)
                        sleep = record.value['sleep']
                        consumer.close() # 避免offset没提交，阻塞其他消费者获取任务
                        time.sleep(sleep)
                print('--------poll index is {} {}----------'.format(index, datetime.now().isoformat()))

    def recv_seek(self, offset):
        """
         接收消费者中的数据,按照 offset 的指定消费位置;
         Args:
           offset: int; kafka 消费者中指定的消费位置;
    
         Returns:
           generator; 消费者消息的生成器;
         """

        self.consumer.seek(self.topic, offset)
        for message in self.consumer:
                # print("主题:%s 分区:%d:连续值:%d: 键:key=%s 值:value=%s" % (
                #   message.topic, message.partition, message.offset, message.key, message.value))
              yield {"topic": message.topic, "partition": message.partition, "key": message.key,
                  "value": message.value.decode('utf-8')}
        
        
    @staticmethod
    def get_consumer(group_id: str, bootstrap_servers: list, topic: str, enable_auto_commit=True) -> KafkaConsumer:
        if enable_auto_commit:
            return KafkaConsumer(
                topic,
                group_id=group_id,
                bootstrap_servers=bootstrap_servers,
                auto_offset_reset='earliest',
                # fetch_max_bytes=FETCH_MAX_BYTES,
                # connections_max_idle_ms=CONNECTIONS_MAX_IDLE_MS,
                # max_poll_interval_ms=KAFKA_MAX_POLL_INTERVAL_MS,
                # session_timeout_ms=SESSION_TIMEOUT_MS,
                # max_poll_records=KAFKA_MAX_POLL_RECORDS,
                # request_timeout_ms=REQUEST_TIMEOUT_MS,
                # auto_commit_interval_ms=AUTO_COMMIT_INTERVAL_MS,
                value_deserializer=lambda m: json.loads(m.decode('utf-8'))
            )
        else:
            return KafkaConsumer(
                topic,
                group_id=group_id,
                bootstrap_servers=bootstrap_servers,
                auto_offset_reset='earliest',
                # fetch_max_bytes=FETCH_MAX_BYTES,
                # connections_max_idle_ms=CONNECTIONS_MAX_IDLE_MS,
                # max_poll_interval_ms=KAFKA_MAX_POLL_INTERVAL_MS,
                # session_timeout_ms=SESSION_TIMEOUT_MS,
                # max_poll_records=KAFKA_MAX_POLL_RECORDS,
                # request_timeout_ms=REQUEST_TIMEOUT_MS,
                enable_auto_commit=enable_auto_commit,
                value_deserializer=lambda m: json.loads(m.decode('utf-8'))
            )
    # auto_offset_reset ：消费者启动的时刻，消息队列中或许已经有堆积的未消费消息，有时候需求是从上一次未消费的位置开始读(则该参数设置为 earliest )，有时候的需求为从当前时刻开始读之后产生的，之前产生的数据不再消费(则该参数设置为 latest )。
    # enable_auto_commit， auto_commit_interval_ms ：是否自动commit，当前消费者消费完该数据后，需要commit，才可以将消费完的信息传回消息队列的控制中心。enable_auto_commit 设置为 True 后，消费者将自动 commit，并且两次 commit 的时间间隔为 auto_commit_interval_ms 。

    # 查看 kafka 堆积剩余量
    #         在线环境中，需要保证消费者的消费速度大于生产者的生产速度，所以需要检测 kafka 中的剩余堆积量是在增加还是减小。可以用如下代码，观测队列消息剩余量：
    #
    # consumer = KafkaConsumer(topic, **kwargs)
    # partitions = [TopicPartition(topic, p) for p in consumer.partitions_for_topic(topic)]
    #
    # print("start to cal offset:")
    #
    # # total
    # toff = consumer.end_offsets(partitions)
    # toff = [(key.partition, toff[key]) for key in toff.keys()]
    # toff.sort()
    # print("total offset: {}".format(str(toff)))
    #
    # # current
    # coff = [(x.partition, consumer.committed(x)) for x in partitions]
    # coff.sort()
    # print("current offset: {}".format(str(coff)))
    #
    # # cal sum and left
    # toff_sum = sum([x[1] for x in toff])
    # cur_sum = sum([x[1] for x in coff if x[1] is not None])
    # left_sum = toff_sum - cur_sum
    # print("kafka left: {}".format(left_sum))
    
def main():
    con = Consumer()
    con.test_poll()
    # con.recv()

if __name__ == "__main__":
    main()
