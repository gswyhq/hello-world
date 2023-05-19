#!/usr/bin/python
# -*- coding:utf8 -*- #
from __future__ import generators
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import re
import time
import random
from datetime import datetime, timedelta, date
from functools import wraps

import logging

def re_compile(pattern, ignore=False):
    """对输入的正则表达式进行预编译
    ignore为真时，则删除正则表达式首位的'^$'进行匹配
    """

    def ignore_compile_pattern(pattern):
        if pattern.startswith("^") and pattern.endswith("$"):
            return re.compile(pattern[1:-1])
        elif pattern.startswith("^"):
            return re.compile(pattern[1:])
        elif pattern.endswith("$"):
            return re.compile(pattern[:-1])
        else:
            return re.compile(pattern)

    if isinstance(pattern, list):
        if ignore:
            return [ignore_compile_pattern(t) for t in pattern]
        return [re.compile(t) for t in pattern]
    if ignore:
        return ignore_compile_pattern(pattern)
    return re.compile(pattern)


def timing(f):
    """函数运行时间的计时器"""

    @wraps(f)  # 保留原有函数的名称和docstring
    def wrap(*args, **kwargs):
        time1 = time.time()
        ret = f(*args, **kwargs)
        time2 = time.time()
        logging.info('<函数名称: {0}>;[运行耗时: {1} ms]'.format(f.__name__, (time2 - time1) * 1000.0))
        return ret

    return wrap


def random_probability(probability=0.3):
    """事件发生的概率probability，返回真假"""
    if random.random() < probability:
        # random.random()用于生成一个0到1的随机符点数: 0 <= n < 1.0
        return True
    else:
        return False

def is_period_hour_and_minute(start_hour=7, start_minute=0, end_hour=9, end_minute=0):
    """
    判断当前时间是不是某个时间范围(单位：小时:分钟)，若开始时间大于结束时间，则认为结束时间是次日
    例一：判断当前时间是否是7:00~10:30
        is_period_hour_and_minute(7,0,10,30)
    例二：判断当前时间是否是晚上十点到次日凌晨5点
      is_period_hour_and_minute(22,0,5,0)
    若是，返回真，否则返回假
    :param start_hour: 开始的时间，时，有效范围0~23
    :param start_minute: 开始的时间，分，有效范围0~59
    :param end_hour: 结束的时间，时，有效范围0~23
    :param end_minute: 结束的时间，分，有效范围0~59
    :return: True or False
    """
    starttime = datetime.now().replace(hour=start_hour, minute=start_minute)
    nowtime = datetime.now()
    endtime = datetime.now().replace(hour=end_hour, minute=end_minute)

    if start_hour > end_hour:
        critical_end_time = (nowtime + timedelta(days=1)).replace(hour=0, minute=0)  # 次日零点
        critical_start_time = nowtime.replace(hour=0, minute=0)  # 今日零点
        return (nowtime >= starttime and nowtime < critical_end_time) or (nowtime >= critical_start_time and nowtime < endtime)
    elif nowtime >= starttime and nowtime < endtime:
        return True
    else:
        return False

# 将字典转换为对象，类似属性访问:
config = {'architectures': ['BertForMaskedLM'],
 'attention_probs_dropout_prob': 0.1,
 'bos_token_id': 0,
 'directionality': 'bidi',
 'eos_token_id': 2,
 'hidden_act': 'gelu',
 'hidden_dropout_prob': 0.1,
 'hidden_size': 768,
 'biaffine_size': 256,
 'initializer_range': 0.02,
 'intermediate_size': 3072,
}
class Dict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__

def dict_to_object(dictObj):
    if not isinstance(dictObj, dict):
        return dictObj
    inst = Dict()
    for k, v in dictObj.items():
        inst[k] = dict_to_object(v)
    return inst

# 转换字典成为对象，可以用"."方式访问对象属性
args = dict_to_object(config)

def main():
    pass


if __name__ == "__main__":
    main()
