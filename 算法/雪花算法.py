#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 使用一种简单一些的ID，并且希望ID能够按照时间有序生成。
# twitter的snowflake解决了这种需求，snowflake的结构如下(每部分用-分开):
# 0 - 0000000000 0000000000 0000000000 0000000000 0 - 00000 - 00000 - 000000000000
# 第一位为未使用，接下来的41位为毫秒级时间(41位的长度可以使用69年)，然后是5位datacenterId和5位workerId(10位的长度最多支持部署1024个节点） ，最后12位是毫秒内的计数（12位的计数顺序号支持每个节点每毫秒产生4096个ID序号）
# 一共加起来刚好64位，为一个Long型。(转换成字符串长度为18)
# 使用雪花算法生成的主键，二进制表示形式包含 4 部分，从高位到低位分表为：1bit 符号位、41bit 时间戳位、10bit 工作进程位以及 12bit 序列号位。
# 符号位（1bit）
# 预留的符号位，恒为零。
# 时间戳位（41bit）
# 41 位的时间戳可以容纳的毫秒数是 2 的 41 次幂，一年所使用的毫秒数是：365 * 24 * 60 * 60 * 1000。通过计算可知：
# Math.pow(2, 41) / (365 * 24 * 60 * 60 * 1000L);
# 结果约等于 69.73 年。


import socket
import time


class IdWorker(object):
    # 获取主机名
    HOSTNAME = socket.gethostname()
    # 获取IP
    IP = socket.gethostbyname(HOSTNAME)
    # 序列号(12位  0-4095)
    SERIAL_NUMBER = 0
    # 时间戳（41位）
    TIMESTAMP = int(time.time() * 1e3)
    # 机器id(10位  0-1024)  这里取机器ip最后8位
    MACHINE_ID = int(IP.split('.')[3])

    @classmethod
    def generate(cls):
        now = int(time.time() * 1e3)
        if now == cls.TIMESTAMP:
            cls.SERIAL_NUMBER += 1
            if cls.SERIAL_NUMBER >= 2**12:
                # 该序列是用来在同一个毫秒内生成不同的 ID。如果在这个毫秒内生成的数量超过 4096 (2的12次幂)，那么生成器会等待到下个毫秒继续生成。
                time.sleep(1/1e3)
                return cls.generate()
        else:
            cls.TIMESTAMP = now
            cls.SERIAL_NUMBER = 0
        return (cls.TIMESTAMP << 22) + (cls.MACHINE_ID << 12) + cls.SERIAL_NUMBER


def ip_to_addr():
    # 附：IP地址和整型互相转换
    IP = '192.168.1.117'
    IP_number = sum(int(j) << i * 8 for i, j in enumerate(IP.split('.')[::-1]))
    print(IP_number)
    IP_addr = '.'.join([str((IP_number // 256 ** i) % 256) for i in range(3, -1, -1)])
    print(IP_addr )

def main():
    import random

    for i in range(100):
        # time.sleep(random.random())
        _id = IdWorker.generate()
        print(_id)

if __name__ == '__main__':
    main()