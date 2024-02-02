#!/usr/bin/env python
# coding=utf-8

import json
import traceback
import random
import kafka
from kafka import KafkaClient, KafkaAdminClient
from kafka.admin import NewTopic
from kafka.errors import KafkaError
from config import cfg
from utils.write_logs_utils import logger

bootstrap_servers = ['192.168.3.105:9092']
topic = 'test3'

NUM_PARTITIONS = 50

class Producer():
    def __init__(self, topic=SDMP_DMA_TOPIC, bootstrap_servers=SDMP_BOOTSTRAP_SERVERS_LIST, api_version=(1, 0, 0)):
        self.consumer = kafka.KafkaConsumer(bootstrap_servers=bootstrap_servers,
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

        self.producer = kafka.KafkaProducer(bootstrap_servers=bootstrap_servers, api_version=api_version,
                                            value_serializer=lambda v: json.dumps(v, ensure_ascii=False).encode(
                                                'utf-8'))
        self.topic = topic

    def sync_producer(self, data_li: list):
        """
        同步发送 数据
        :param data_li:  发送数据
        :return:
        """
        for data in data_li:
            try:
                future = self.producer.send(self.topic, data)
                record_metadata = future.get(timeout=10)  # 同步确认消费
                partition = record_metadata.partition  # 数据所在的分区
                offset = record_metadata.offset  # 数据所在分区的位置
                print('kafka发送消息成功,topic: {}, partition: {}, offset: {}'.format(record_metadata.topic, partition, offset))
            except KafkaError as e:
                print("kafka发送消息失败：{}，错误详情：{}", e, traceback.format_exc())

    def asyn_producer(self, data_li: list):
        """
        异步发送数据
        :param data_li:发送数据
        :return:
        """
        try:
            for data in data_li:
                self.producer.send(self.topic, data)
            self.producer.flush()  # 批量提交
            print("批量提交完成！")
        except Exception as e:
            print("提交失败：{}，失败详情：{}".format(e, traceback.format_exc()))

    def asyn_producer_callback(self, data_li: list):
        """
        异步发送数据 + 发送状态处理
        :param data_li:发送数据
        :return:
        """
        for data in data_li:
            self.producer.send(self.topic, data).add_callback(self.send_success, data=data).add_errback(self.send_error, data=data)
        self.producer.flush()  # 批量提交

    def send_success(self, *args, **kwargs):
        """异步发送成功回调函数"""
        print('kafka消息推送成功!')
        return

    def send_error(self, *args, **kwargs):
        """异步发送错误回调函数"""
        data = kwargs.get('data', {})
        print(f'kafka消息推送失败, {data}')
        return

    def close_producer(self):
        try:
            self.producer.close()
        except:
            pass

def main():
    con = Producer()
    data_list = [
        {'name': "张三", "key": "asyn", "sleep": 1 },
        {'name': "历史", "key": "asyn", "sleep": 1 },
        {'name': "网上", "key": "asyn", "sleep": 60 },
        {'name': "语义", "key": "asyn", "sleep": 1 },
    ]
    con.asyn_producer(data_list)
    data_list = [
        {'name': "张三", "key": "sync", "sleep": 1 },
        {'name': "历史", "key": "sync", "sleep": 1 },
        {'name': "网上", "key": "sync", "sleep": 60 },
        {'name': "语义", "key": "sync", "sleep": 1 },
    ]
    con.sync_producer(data_list)

    data_list2 = []
    for d in data_list:
        d['key'] = 'asyn_call'
        d['sleep'] = 1
        data_list2.append(d)
    con.asyn_producer_callback(data_list)


if __name__ == "__main__":
    main()
