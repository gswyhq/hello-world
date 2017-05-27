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

# http://www.cnblogs.com/quijote/p/4385774.html
# APScheduler 3.0.1浅析

################################################################################

import time
from threading import Timer

def print_time( enter_time ):
    print ("现在时间是：", time.time() , "输入参数：", enter_time)

print (time.time())
Timer(5,  print_time, ( "参数1", )).start() # 参数后面的逗号不能少
Timer(10, print_time, ( "参数2", )).start()
print (time.time()) # 并不等上面两个执行
#这样的话，从程序开始执行到5,秒，10秒都会执行一次print_time这个方法。

################################################################################
若是CPU密集型，需考虑用ProcessPoolExecutor 代替 ThreadPoolExecutor
APScheduler带有三个内置触发类型：
• date: 当你想在某个时间点刚好运行一次工作中使用
• interval: 当你想在固定的时间间隔运行作业使用
• cron: use 当你想在一天中的特定时间（s）定期运行工作中使用

from apscheduler.schedulers.background import BackgroundScheduler
scheduler = BackgroundScheduler()
# Initialize the rest of the application here, or before the scheduler initialization
# 这里默认使用MemoryJobStore和ThreadPoolExecutor（默认最大线程数为10）

# 下面三个实例完全是等价的
实例1：
from pytz import utc
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.jobstores.mongodb import MongoDBJobStore
# from apscheduler.jobstores.redis import RedisJobStore
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.executors.pool import ThreadPoolExecutor, ProcessPoolExecutor
jobstores = {
            'mongo': MongoDBJobStore(),
            'default': SQLAlchemyJobStore(url='sqlite:///jobs.sqlite')
            }
executors = {
            'default': ThreadPoolExecutor(20),
            'processpool': ProcessPoolExecutor(5)
            }
job_defaults = {
            'coalesce': False,
            'max_instances': 3
            }
scheduler = BackgroundScheduler(jobstores=jobstores, executors=executors, job_defaults=job_defaults, timezone=utc)

实例2：
from apscheduler.schedulers.background import BackgroundScheduler
# The "apscheduler." prefix is hard coded
scheduler = BackgroundScheduler({
                                'apscheduler.jobstores.mongo': {
                                    'type': 'mongodb'
                                    },
                                'apscheduler.jobstores.default': {
                                    'type': 'sqlalchemy',
                                    'url': 'sqlite:///jobs.sqlite'
                                    },
                                'apscheduler.executors.default': {
                                    'class': 'apscheduler.executors.pool:ThreadPoolExecutor',
                                    'max_workers': '20'
                                    },
                                'apscheduler.executors.processpool': {
                                    'type': 'processpool',
                                    'max_workers': '5'
                                    },
                                'apscheduler.job_defaults.coalesce': 'false',
                                'apscheduler.job_defaults.max_instances': '3',
                                'apscheduler.timezone': 'UTC',
                                }
                                )
                                
实例3：
from pytz import utc
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.executors.pool import ProcessPoolExecutor
jobstores = {
    'mongo': {'type': 'mongodb'},
    'default': SQLAlchemyJobStore(url='sqlite:///jobs.sqlite')
    }
executors = {
    'default': {'type': 'threadpool', 'max_workers': 20},
    'processpool': ProcessPoolExecutor(max_workers=5)
    }
job_defaults = {
    'coalesce': False,
    'max_instances': 3
    }
scheduler = BackgroundScheduler()
# .. do something else here, maybe add jobs etc.
scheduler.configure(jobstores=jobstores, executors=executors, job_defaults=job_defaults, timezone=utc)


# 删除作业有两种方法：
# 1.通过调用该作业的ID和作业存储别名remove_job（）
# 2.您从add_job得到了这份工作实例调用remove（）
job = scheduler.add_job(myfunc, 'interval', minutes=2)
job.remove()


    base
    memory
    mongodb
    redis
    rethinkdb
    sqlalchemy


APS_SCHEDULER_CONFIG = {
    # 添加了一个默认(名叫default)的jobstore，它的具体实现类型是sqlalchemy，数据库连接url是指向一个本地postgresql数据库，也就是说添加到这个scheduler的job会默认使用这个jobstore进行存储
    'jobstores': {
        'redis':{'type':'redis'}, # 此项并无效果，可能与安装的方法有关，不应该pip install apscheduler,而是应该python setup.py install
        'default': {'type': 'sqlalchemy', 'url': 'sqlite:///jobs.sqlite'},

        },
    'executors': {
         # 'default': {'type': 'processpool', 'max_workers': 10},
        'default': {'class': 'threadpool', 'max_workers': 10},
        'processpool': {'type': 'processpool', 'max_workers': 10},
        },
    'job_defaults': {
         'coalesce': True,
         'max_instances': 5,
         'misfire_grace_time': 30
        },
    'timezone': 'Asia/Shanghai'
    }
def my_job(text=''):
    print ('你好:{},现在时间：{}'.format(text,datetime.now()))

class TimeingTask():
    """定时任务"""
    def __init__(self):
        # self.scheduler = BackgroundScheduler(daemonic = True) # daemonic = False,
        # self.scheduler = BackgroundScheduler(jobstores=jobstores, executors=executors, job_defaults=job_defaults, timezone=pytz.timezone('Asia/Shanghai'))
        self.scheduler = BackgroundScheduler(APS_SCHEDULER_CONFIG)
        
        
        

def main():
    pass


if __name__ == "__main__":
    main()
