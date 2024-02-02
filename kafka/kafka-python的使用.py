#!/usr/bin/env python
# coding=utf-8

# pip3 install kafka-python
import time, json
import kafka
from kafka import KafkaClient, KafkaProducer, KafkaConsumer, KafkaAdminClient, TopicPartition
from kafka.admin import NewTopic

bootstrap_servers = ['192.168.3.105:9092']
api_version=(1, 1, 0)

###########################################################################################################################
# 查询有哪些主题
consumer = kafka.KafkaConsumer(bootstrap_servers=bootstrap_servers, api_version=api_version,
                                            auto_offset_reset='earliest',
                                            enable_auto_commit=True,
                                            value_deserializer=lambda m: json.loads(m.decode('utf-8'))
                                            )
topics = consumer.topics()
print("当前主题列表：{}".format(topics))

client = KafkaClient(bootstrap_servers=bootstrap_servers, api_version=api_version)
if topic not in client.cluster.topics(exclude_internal_topics=True):  # Topic不存在, 但貌似这种情况判断有时候判断不对
    print('当前主题不存在')

###########################################################################################################################
# 删除主题,这里是删除主题test_topic
admin = KafkaAdminClient(bootstrap_servers=bootstrap_servers, api_version=api_version)
admin.delete_topics(['test_topic'])

###########################################################################################################################
# 创建一个主题
new_topic = NewTopic('test3', num_partitions=10, replication_factor=1)
admin.create_topics([new_topic])

###########################################################################################################################
# 查询主题下有多少个分区
partitions = consumer.partitions_for_topic('test3')

###########################################################################################################################
# 使用kafka-python,同样可以获取统计到总数：
consumer = KafkaConsumer(bootstrap_servers=bootstrap_servers)
topic = 'test'
# 获取topic的分区
partition = list(consumer.partitions_for_topic(topic=topic))[0]
partitions = list(consumer.partitions_for_topic(topic=topic))

sum = 0
for partition in partitions:
    # print(partition)
    topic_partition = TopicPartition(topic=topic, partition=partition)
    consumer.assign([topic_partition])
    start_offset = consumer.beginning_offsets([topic_partition])[topic_partition]
    end_offset = consumer.end_offsets([topic_partition])[topic_partition]
    a = end_offset - start_offset
    print('分区：{}，消息数：{}'.format(partition, a))
    sum = sum + a
print("各个分区消息总量：；{}".format(sum))


def main():
    pass


if __name__ == "__main__":
    main()
