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

#一，当你想让你某个方法在几秒甚至更长的时间内执行后执行一次，你可以这样做：

import time
from threading import Timer

def print_time( enter_time ):
    print ("现在时间是：", time.time() , "输入参数时间：", enter_time)


print (time.time())
Timer(5,  print_time, ( time.time(), )).start()
Timer(10, print_time, ( time.time(), )).start()
print (time.time())
#这样的话，从程序开始执行到5,秒，10秒都会执行一次print_time这个方法。

# 当你想让你的某个方法每个一定周期执行呢，这就需要定时任务的框架APScheduler，安装方法：sudo pip2 install apscheduler

# 1，带修饰器的写法：

from apscheduler.scheduler import Scheduler
import datetime
schedudler = Scheduler(daemonic = False)

@schedudler.cron_schedule(second='15', day_of_week='0-7', hour='9-12,13-16')
def quote_send_sh_job():
    print 'a simple cron job start at', datetime.datetime.now()

schedudler.start()

# 2，不带修饰器的写法：

def cornstart(self,event):
         schedudler = Scheduler(daemonic = True)

         schedudler.add_cron_job(self.timing_exe, day_of_week='mon-sun', hour='0-12', minute='0-59', second='15',)
         print 'get start'
         schedudler.start()

#应该好理解，timing_exe是要执行的函数名，如果函数还有参数可以加一个args[]

以默认配置启动Scheduler


from apscheduler.scheduler import Scheduler
   
sched = Scheduler()
sched.start()
1.基于固定时间的调度：


from datetime import date
from apscheduler.scheduler import Scheduler
   
# 启动Scheduler
sched = Scheduler()
sched.start()
   
# 定义被自动调度的函数
def my_job(text):
    print text
   
# 定义任务的执行时间（2013年5月17日）
exec_date = date(2013, 5, 17)
   
# 加入到任务队列，并将其赋值给变量以方便取消等操作
job = sched.add_date_job(my_job, exec_date, ['text'])

2.周期任务：

def job_function():
    print "Hello World"
    
# job_function将会每两小时执行一次
sched.add_interval_job(job_function, hours=2)
    
# 与上面的任务相同，不过规定在2013-5-17 18:30之后才开始运行
sched.add_interval_job(job_function, hours=2, start_date='2013-5-17 18:30')
   装饰器版本：

@sched.interval_schedule(hours=2)
def job_function():
    print "Hello World"
3.Cron风格的任务的调度：


def job_function():
    print "Hello World"
   
# 安排job_function函数将会在六月、七月、十一月和十二月的第三个星期五中的0点、1点、2点和3点分别执行
sched.add_cron_job(job_function, month='6-8,11-12', day='3rd fri', hour='0-3')
 装饰器版本：

@sched.cron_schedule(day='last sun')
def some_decorated_task():
    print "I am printed at 00:00:00 on the last Sunday of every month!"
更强大的使用方法可以进一步参考官方文档。


################################################################################
from apscheduler.scheduler import Scheduler  
  
schedudler = Scheduler(daemonic = False)  
 
@schedudler.cron_schedule(second='*', day_of_week='0-4', hour='9-12,13-15')  
def quote_send_sh_job():  
    print 'a simple cron job start at', datetime.datetime.now()  
  
schedudler.start() 
上面通过装饰器定义了cron job，可以通过函数scheduler.add_cron_job添加，用装饰器更方便。Scheduler构造函数中传入daemonic参数，表示执行线程是非守护的，在Schduler的文档中推荐使用非守护线程

在添加job时还有一个比较重要的参数max_instances，指定一个job的并发实例数，默认值是1。默认情况下，如果一个job准备执行，但是该job的前一个实例尚未执行完，则后一个job会失败，可以通过这个参数来改变这种情况。

APScheduler提供了jobstore用于存储job的执行信息，默认使用的是RAMJobStore，还提供了SQLAlchemyJobStore、ShelveJobStore和MongoDBJobStore。APScheduler允许同时使用多个jobstore，通过别名（alias）区分，在添加job时需要指定具体的jobstore的别名，否则使用的是别名是default的jobstore，即RAMJobStore。下面以MongoDBJobStore举例说明。
[python] view plain copy print?
import pymongo  
from apscheduler.scheduler import Scheduler  
from apscheduler.jobstores.mongodb_store import MongoDBJobStore  
import time  
  
sched = Scheduler(daemonic = False)  
  
mongo = pymongo.Connection(host='127.0.0.1', port=27017)  
store = MongoDBJobStore(connection=mongo)  
sched.add_jobstore(store, 'mongo')        # 别名是mongo  

@sched.cron_schedule(second='*', day_of_week='0-4', hour='9-12,13-15', jobstore='mongo')        # 向别名为mongo的jobstore添加job  
def job():  
        print 'a job'  
        time.sleep(1)  
  
sched.start()  
        注意start必须在添加job动作之后调用，否则会抛错。默认会把job信息保存在apscheduler数据库下的jobs表：
[plain] view plain copy print?
> db.jobs.findOne()  
{  
        "_id" : ObjectId("502202d1443c1557fa8b8d66"),  
        "runs" : 20,  
        "name" : "job",  
        "misfire_grace_time" : 1,  
        "coalesce" : true,  
        "args" : BinData(0,"gAJdcQEu"),  
        "next_run_time" : ISODate("2012-08-08T14:10:46Z"),  
        "max_instances" : 1,  
        "max_runs" : null,  
        "trigger" : BinData(0,"gAJjYXBzY2hlZHVsZXIudHJpZ2dlcnMuY3JvbgpDcm9uVHJpZ2dlcgpxASmBcQJ9cQMoVQZmaWVsZHNxBF1xBShjYXBzY2hlZHVsZXIudHJpZ2dlcnMuY3Jvbi5maWVsZHMKQmFzZUZpZWxkCnEGKYFxB31xCChVCmlzX2RlZmF1bHRxCYhVC2V4cHJlc3Npb25zcQpdcQtjYXBzY2hlZHVsZXIudHJpZ2dlcnMuY3Jvbi5leHByZXNzaW9ucwpBbGxFeHByZXNzaW9uCnEMKYFxDX1xDlUEc3RlcHEPTnNiYVUEbmFtZXEQVQR5ZWFycRF1YmgGKYFxEn1xEyhoCYhoCl1xFGgMKYFxFX1xFmgPTnNiYWgQVQVtb250aHEXdWJjYXBzY2hlZHVsZXIudHJpZ2dlcnMuY3Jvbi5maWVsZHMKRGF5T2ZNb250aEZpZWxkCnEYKYFxGX1xGihoCYhoCl1xG2gMKYFxHH1xHWgPTnNiYWgQVQNkYXlxHnViY2Fwc2NoZWR1bGVyLnRyaWdnZXJzLmNyb24uZmllbGRzCldlZWtGaWVsZApxHymBcSB9cSEoaAmIaApdcSJoDCmBcSN9cSRoD05zYmFoEFUEd2Vla3EldWJjYXBzY2hlZHVsZXIudHJpZ2dlcnMuY3Jvbi5maWVsZHMKRGF5T2ZXZWVrRmllbGQKcSYpgXEnfXEoKGgJiWgKXXEpY2Fwc2NoZWR1bGVyLnRyaWdnZXJzLmNyb24uZXhwcmVzc2lvbnMKUmFuZ2VFeHByZXNzaW9uCnEqKYFxK31xLChoD05VBGxhc3RxLUsEVQVmaXJzdHEuSwB1YmFoEFULZGF5X29mX3dlZWtxL3ViaAYpgXEwfXExKGgJiWgKXXEyKGgqKYFxM31xNChoD05oLUsMaC5LCXViaCopgXE1fXE2KGgPTmgtSw9oLksNdWJlaBBVBGhvdXJxN3ViaAYpgXE4fXE5KGgJiGgKXXE6aAwpgXE7fXE8aA9Oc2JhaBBVBm1pbnV0ZXE9dWJoBimBcT59cT8oaAmJaApdcUBoDCmBcUF9cUJoD05zYmFoEFUGc2Vjb25kcUN1YmVVCnN0YXJ0X2RhdGVxRE51Yi4="),  
        "func_ref" : "__main__:job",  
        "kwargs" : BinData(0,"gAJ9cQEu")  
}  
        上面就是存储的具体信息。
最后，需要注意一点当job不以daemon模式运行时，并且APScheduler也不是daemon的，那么在关闭脚本时，Ctrl + C是不奏效的，必须kill才可以。可以通过命令实现关闭脚本：



def main():
    pass


if __name__ == "__main__":
    main()
