#!/usr/bin/env python
# coding=utf-8

import time, json
import kafka

bootstrap_servers = ['192.168.3.105:9092']
topic = 'test3'
group_id = 't1234'

consumer = kafka.KafkaConsumer(topic, group_id=group_id,
                               bootstrap_servers=bootstrap_servers, api_version=(1, 1, 0),
                               auto_offset_reset='earliest',
                               enable_auto_commit=True, # 自动提交消费数据的offset
                               value_deserializer=lambda m: json.loads(m.decode('utf-8'))
                               )
index = 0
while True:
    msg = consumer.poll(timeout_ms=5, max_records=1)  # 从kafka获取max_records条消息

    time.sleep(1)
    index += 1
    if msg:
        print(msg)
        for tp, records in msg.items():
            for record in records:
                print(record.offset, record.value)
        print('--------poll index is %s----------' % index)


def main():
    pass


if __name__ == "__main__":
    main()
