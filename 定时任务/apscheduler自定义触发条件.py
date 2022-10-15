#!/usr/bin/python
# -*- coding:utf8 -*- #
from __future__ import  generators
from __future__ import  division
from __future__ import  print_function
from __future__ import  unicode_literals
import sys,os,json

if sys.version >= '3':
    PY3 = True
else:
    PY3 = False

if PY3:
    import pickle
else:
    import cPickle as pickle
    from codecs import open

import time
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from apscheduler.triggers.base import BaseTrigger
"""
最近在使用APScheduler做作业调度的时候碰到一个问题，就是有时候我的作业用一个cron trigger不能完全包含所有的触发条件，同时这个作业不能被创建多个实例。

比如期望作业在下面时间运行

每周一到周五的09:00-18:00
每周六，周日的00:00-24:00
…
此时我们可能就不能仅仅通过一个cron来描述我所有的触发条件，对于这种问题，我期望能对一个作业指定多个cron trigger，当任何一个trigger满足时就触发作业允许。

看了一下APScheduler，它可以支持用户自定trigger，实现起来也比较简单，就是要继承apscheduler.triggers.base.BaseTrigger类，并实现get_next_fire_time(previous_fire_time, now)方法。

好了，说干就干，先实现我的MultiCronTrigger类

这个类里主要包含一个triggers列表，保存所有CronTrigger对象。
get_next_fire_time(self, previous_fire_time, now)方法遍历所有CronTrigger对象，调用每个CronTrigger对象的get_next_fire_time()方法，找到其中最小的一个时间然后返回。如果triggers是空或者没有找到就返回None。
理论上这个类是不仅仅只支持传递多个CronTrigger对象，其实它也可以支持传递多个IntervalTrigger对象，或者混合两种对象，但感觉对我意义不大，所以这里主要说传递多个CronTrigger对象的情况。

整个测试类试试，测试类定义了两个CronTrigger对象，第一个trigger每三秒运行一次作业，第二个trigger每五秒运行一次作业。

运行一下测试，可以看到job在每个3的倍数和5的倍数秒上都会执行作业。
"""

class MultiCronTrigger(BaseTrigger):
    """用户自定trigger"""
    triggers = []

    def __init__(self, triggers):
        self.triggers = triggers

    def get_next_fire_time(self, previous_fire_time, now):
        min_next_fire_time = None
        for trigger in self.triggers:
            next_fire_time = trigger.get_next_fire_time(previous_fire_time, now)
            if next_fire_time is None:
                continue
            if min_next_fire_time is None:
                min_next_fire_time = next_fire_time
            if next_fire_time < min_next_fire_time:
                min_next_fire_time = next_fire_time
        return min_next_fire_time
        
        
def myjob():
    print('job run at %s' % datetime.now())

if __name__ == '__main__':
    scheduler = BackgroundScheduler()
    trigger1 = CronTrigger(hour='*', minute='*', second='*/3') # 第一个trigger每三秒运行一次作业
    trigger2 = CronTrigger(hour='*', minute='*', second='*/5') # 第二个trigger每五秒运行一次作业。
    job = scheduler.add_job(myjob, MultiCronTrigger([trigger1, trigger2]))
    scheduler.start()

try:
    while True:
        time.sleep(5)
except (KeyboardInterrupt, SystemExit):
    scheduler.shutdown()
    
    
