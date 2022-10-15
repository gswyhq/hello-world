#!/usr/bin/python
# -*- coding:utf8 -*- #
from __future__ import  generators
from __future__ import  division
from __future__ import  print_function
from __future__ import  unicode_literals
import sys,os,json
import time
from datetime import datetime
if sys.version >= '3':
    PY3 = True
else:
    PY3 = False

if PY3:
    import pickle
else:
    import cPickle as pickle
    from codecs import open

from apscheduler.schedulers.blocking import BlockingScheduler

def my_job(text):
    print ('你好:{}'.format(text))

def tick():
    print('第二个作业! 现在时间是: %s' % datetime.now())
 
def test1():
    """定时执行任务"""
    start_time = time.time()
    sched = BlockingScheduler()
    sched.add_job(my_job, 'interval', args=('123',),seconds=1, id='my_job_id') # 每隔1秒执行一次my_job函数,args为函数my_job的输入参数；id:可省略；
    sched.start() # 程序运行到这里，并不往后运行，除非把任务都完成，但ctrl+C 可以终止
    print('运行不到这里')

def test5():
    """定时执行任务，关闭调度器"""
    sched = BlockingScheduler()
    sched.add_job(my_job, 'date',run_date=datetime(2016, 8, 16, 12, 34,5), args=('123',),seconds=1, id='my_job_id') 
    # add_job的第二个参数是trigger，它管理着作业的调度方式。它可以为date, interval或者cron。
    sched.start()
    print('定时任务')
    
from apscheduler.schedulers.background import BackgroundScheduler

def test2():
    """定时执行任务，暂停，恢复"""
    start_time = time.time()
    scheduler = BackgroundScheduler(daemonic = False) # daemonic = False,
    scheduler.add_job(my_job, 'interval', args=('123',),seconds=1, id='my_job_id') # 每隔1秒执行一次my_job函数,args为函数my_job的输入参数；id:可省略；
    scheduler.start() # 程序运行到这里，任务没有运行完也会往后执行，既执行后面的任务，又执行这个任务。
    print('运行到了这里1')
    while (scheduler.state):
        if time.time() - start_time >5:
            print('暂停作业')
            scheduler.pause() # 暂停作业:
            break
    print('恢复作业')
    if time.time() - start_time >5:
        scheduler.resume() # 恢复作业
    time.sleep(4)
    print('再次暂停作业')
    #scheduler.pause() # 暂停作业,进程并未被守护，即使没有这句也是同样的效果
    #return scheduler

def test3():
    """定时执行任务，暂停，恢复"""
    start_time = time.time()
    scheduler = BackgroundScheduler()
    scheduler.add_job(my_job, 'interval', args=('123',),seconds=1, id='my_job_id') # 每隔1秒执行一次my_job函数,args为函数my_job的输入参数；id:可省略；
    scheduler.start() # 程序运行到这里，任务没有运行完也会往后执行，既执行后面的任务，又执行这个任务。
    print('运行到了这里1')
    while (scheduler.state):
        if time.time() - start_time >5:
            print('暂停作业')
            scheduler.pause() # 暂停作业:
            break
    print('恢复作业')
    if time.time() - start_time >5:
        scheduler.resume() # 
    time.sleep(4)
    print('当前任务列表：{}'.format(scheduler.get_jobs())) # 获得调度作业的列表，可以使用 get_jobs() 来完成，它会返回所有的job实例
    scheduler.get_job('my_job_id') # 获取id为my_job_id的作业实例
    
    scheduler.print_jobs() # 输出所有格式化的作业列表。
    
    print('移除作业')
    # scheduler.remove_job('my_job_id') # 移除id为my_job_id的作业
    scheduler.remove_all_jobs() # 移除所有的作业
    #return scheduler #此时scheduler为空，若return则报错

def test6():
    """定时执行任务，暂停，恢复, 实例化"""
    start_time = time.time()
    scheduler = BackgroundScheduler()
    job = scheduler.add_job(my_job, 'interval', args=('123',),seconds=1, id='my_job_id') # 每隔1秒执行一次my_job函数,args为函数my_job的输入参数；id:可省略；
    print("作业id:{}，作业名字：{}，作业参数：{}，作业函数：{}，触发条件：{}".format(job.id, job.name, job.args, job.func,  job.trigger))
    scheduler.start() # 程序运行到这里，任务没有运行完也会往后执行，既执行后面的任务，又执行这个任务。
    print('运行到了这里1')
    while (scheduler.state):
        if time.time() - start_time >5:
            print('暂停作业')
            #scheduler.pause() # 暂停作业:
            job.pause() # 暂停单个实例
            break
    print('恢复作业')
    if time.time() - start_time >5:
        #scheduler.resume() # 恢复作业
        job.resume() # 恢复单个实例
    time.sleep(4)
    print('当前任务列表：{}'.format(scheduler.get_jobs())) # 获得调度作业的列表，可以使用 get_jobs() 来完成，它会返回所有的job实例
    scheduler.get_job('my_job_id') # 获取id为my_job_id的作业实例
    
    scheduler.print_jobs() # 输出所有格式化的作业列表。
    
    print('移除作业')
    # scheduler.remove_job('my_job_id') # 移除id为my_job_id的作业
    # scheduler.remove_all_jobs() # 移除所有的作业
    job.remove() # 移除单个实例的作业
    #return scheduler #此时scheduler为空，若return则报错


def test4():
    """定时执行任务，关闭调度器"""
    start_time = time.time()
    scheduler = BackgroundScheduler()
    scheduler.add_job(my_job, 'interval', args=('123',),seconds=1, id='my_job_id') 
    # 每隔1秒执行一次my_job函数,args为函数my_job的输入参数；id:可省略；
    
    scheduler.start() # 程序运行到这里，任务没有运行完也会往后执行，既执行后面的任务，又执行这个任务。
    print('运行到了这里1')
    
    #默认情况下调度器会等待所有正在运行的作业完成后，关闭所有的调度器和作业存储。如果你不想等待，可以将wait选项设置为False。
    time.sleep(1)
    print('关闭所有的调度器和作业存储')
    #scheduler.shutdown()
    scheduler.shutdown(wait=False)
    print("输出所有格式化的作业列表:{}".format(scheduler.print_jobs())) # 输出所有格式化的作业列表。
    
# 若想要持久化，必须定义作业的id，及 replace_existing=True

def test7():
    """定时执行任务，通过ctrl+c终止"""
    scheduler = BlockingScheduler()
    scheduler.add_job(tick, 'interval', seconds=1)
    print('Press Ctrl+{0} to exit'.format('Break' if os.name == 'nt' else 'C'))
    try:
        scheduler.start(paused=False) # 后面的并没有执行,若参数paused为真,则并不执行
    except (KeyboardInterrupt, SystemExit):
        pass
    print('ok')
 
def test8():
    """定时执行任务，修改作业"""
    scheduler = BackgroundScheduler() 
    job1 = scheduler.add_job(my_job, 'interval', args=('123',),seconds=2, id='my_job_id',max_instances=2) 
    # 每隔1秒执行一次my_job函数,args为函数my_job的输入参数；id:可省略；
    print('开始scheduler')
    scheduler.start()
    time.sleep(2)
    print('再添加一个实例')
    job = scheduler.add_job(tick, 'interval', seconds=2,id='id2') 
    # add_job函数这里有一个max_instances参数用于指定当前工作同时运行的实例数量，
    
    time.sleep(4)
    job.modify(max_instances=6, name='Alternate name') # 上一个工作正在运行而未结束，那么下一个就认为失败, 那么此时可以为一个特定的作业设置最大数量的调度程序;可以为一个特定的作业设置最大数量的调度程序
    # scheduler.reschedule_job('my_job_id', trigger='cron', minute='*/5') # 每5分钟执行一次，具体参见cron表达式
    print('修改执行时间间隔')
    scheduler.reschedule_job('id2', trigger='cron', second='*/1') # 每1秒执行一次，具体参见cron表达式
    time.sleep(3)
    print('ok')

def my_listener(event):
    if event.exception:
        print('The job crashed :(')
    else:
        print('The job worked :)')
            
def test9():
    """异常处理
        当job抛出异常时，APScheduler会默默的把他吞掉，不提供任何提示，这不是一种好的实践，我们必须知晓程序的任何差错。APScheduler提供注册listener，可以监听一些事件，包括：job抛出异常、job没有来得及执行等。
        """

    scheduler.add_listener(my_listener, EVENT_JOB_EXECUTED | EVENT_JOB_ERROR)
     
def test10():
    """固定时间执行"""
    scheduler = BackgroundScheduler()
    


    #scheduler.add_job(my_job, 'date', run_date=datetime(2016, 8, 16, 18, 00, 55), args=['text']) 
    scheduler.add_job(tick, trigger='date', run_date='2016-8-16 17:27:00')  # 与上面的意思差不多
    # add_job的第二个参数是trigger，它管理着作业的调度方式。它可以为date, interval或者cron。对于不同的trigger，对应的参数也相同。
    # date 定时调度;最基本的一种调度，作业只会执行一次。
    scheduler.start()
    
    # 每两个小时执行一次
    # sched.add_job(job_function, 'interval', hours=2) 
    # (1). cron定时调度
    # year (int|str) – 4-digit year
    # month (int|str) – month (1-12)
    # day (int|str) – day of the (1-31)
    # week (int|str) – ISO week (1-53)
    # day_of_week (int|str) – number or name of weekday (0-6 or mon,tue,wed,thu,fri,sat,sun)
    # hour (int|str) – hour (0-23)
    # minute (int|str) – minute (0-59)
    # second (int|str) – second (0-59)
    # start_date (datetime|str) – earliest possible date/time to trigger on (inclusive)
    # end_date (datetime|str) – latest possible date/time to trigger on (inclusive)
    # timezone (datetime.tzinfo|str) – time zone to use for the date/time calculations (defaults to scheduler timezone)

    # (2). interval 间隔调度
    # 它的参数如下：
    # weeks (int) – number of weeks to wait
    # days (int) – number of days to wait
    # hours (int) – number of hours to wait
    # minutes (int) – number of minutes to wait
    # seconds (int) – number of seconds to wait
    # start_date (datetime|str) – starting point for the interval calculation
    # end_date (datetime|str) – latest possible date/time to trigger on
    # timezone (datetime.tzinfo|str) – time zone to use for the date/time calculations

    # Schedules job_function to be run on the third Friday
    # of June, July, August, November and December at 00:00, 01:00, 02:00 and 03:00
    # sched.add_job(job_function, 'cron', month='6-8,11-12', day='3rd fri', hour='0-3')

    # Runs from Monday to Friday at 5:30 (am) until 2014-05-30 00:00:00
    # sched.add_job(job_function, 'cron', day_of_week='mon-fri', hour=5, minute=30, end_date='2014-05-30')

def main():
    # test1()
    test2()
    # test3()
    # test4()
    # test5()
    # test6()
    # test7()
    # test8()
    


if __name__ == "__main__":
    main()
    
    





