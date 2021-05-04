#!/usr/bin/python3
# coding: utf-8

import random

def random_weight(weight_data):
    total = sum(weight_data.values())    # 权重求和
    ra = random.uniform(0, total)   # 在0与权重和之前获取一个随机数
    curr_sum = 0
    ret = None
    keys = list(weight_data.keys())
    keys.sort()
    for k in keys:
        curr_sum += weight_data[k]             # 在遍历中，累加当前权重值
        if ra <= curr_sum:          # 当随机数<=当前权重和时，返回权重key
            ret = k
            break
    return ret


def main():
    weight_data = {'a': 6, 'b': 1, 'c': 3}
    ts = [random_weight(weight_data) for i in range(100000)]
    print({t: ts.count(t) for t in set(ts)})
    # {'a': 60067, 'b': 10115, 'c': 29818}

if __name__ == '__main__':
    main()