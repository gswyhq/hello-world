#! /usr/lib/python3
# -*- coding: utf-8 -*-

#Python3 - 时间处理与定时任务
#! /usr/bin/env python
#coding=utf-8
# 获取今天、昨天和明天的日期
# 引入datetime模块
import datetime
#计算今天的时间
today = datetime.date.today()
#计算昨天的时间
yesterday = today - datetime.timedelta(days = 1)
#计算明天的时间
tomorrow = today + datetime.timedelta(days = 1)
#打印这三个时间
print(yesterday, today, tomorrow)


#! /usr/bin/env python
#coding=utf-8
# 计算上一个的时间
#引入datetime,calendar两个模块
import datetime,calendar

last_friday = datetime.date.today()
oneday = datetime.timedelta(days = 1)

while last_friday.weekday() != calendar.FRIDAY:
    last_friday -= oneday

print(last_friday.strftime('%A, %d-%b-%Y'))

#! /usr/bin/env python
#coding=utf-8
# 借助模运算，可以一次算出需要减去的天数，计算上一个星期五
#同样引入datetime,calendar两个模块
import datetime
import calendar

today = datetime.date.today()
target_day = calendar.FRIDAY
this_day = today.weekday()
delta_to_target = (this_day - target_day) % 7
last_friday = today - datetime.timedelta(days = delta_to_target)

print(last_friday.strftime("%d-%b-%Y"))



#! /usr/bin/env python
#coding=utf-8
# 获取一个列表中的所有歌曲的播放时间之和
import datetime

def total_timer(times):
    td = datetime.timedelta(0)
    duration = sum([datetime.timedelta(minutes = m, seconds = s) for m, s in times], td)
    return duration

times1 = [(2, 36),
          (3, 35),
          (3, 45),
          ]
times2 = [(3, 0),
          (5, 13),
          (4, 12),
          (1, 10),
          ]

assert total_timer(times1) == datetime.timedelta(0, 596)
assert total_timer(times2) == datetime.timedelta(0, 815)

print("Tests passed.\n"
      "First test total: %s\n"
      "Second test total: %s" % (total_timer(times1), total_timer(times2)))

#! /usr/bin/env python
#coding=utf-8
# 以需要的时间间隔执行某个命令

import time, os

def re_exe(cmd, inc = 60):
    while True:
        os.system(cmd);
        time.sleep(inc)

re_exe("echo %time%", 5)


#! /usr/bin/env python
#coding=utf-8
#这里需要引入三个模块
import time, os, sched

# 第一个参数确定任务的时间，返回从某个特定的时间到现在经历的秒数
# 第二个参数以某种人为的方式衡量时间
schedule = sched.scheduler(time.time, time.sleep)

def perform_command(cmd, inc):
    os.system(cmd)

def timming_exe(cmd, inc = 60):
    # enter用来安排某事件的发生时间，从现在起第n秒开始启动
    schedule.enter(inc, 0, perform_command, (cmd, inc))
    # 持续运行，直到计划时间队列变成空为止
    schedule.run()


print("show time after 10 seconds:")
timming_exe("echo %time%", 10)


#! /usr/bin/env python
#coding=utf-8
import time, os, sched

# 第一个参数确定任务的时间，返回从某个特定的时间到现在经历的秒数
# 第二个参数以某种人为的方式衡量时间
schedule = sched.scheduler(time.time, time.sleep)

def perform_command(cmd, inc):
    # 安排inc秒后再次运行自己，即周期运行
    schedule.enter(inc, 0, perform_command, (cmd, inc))
    os.system(cmd)

def timming_exe(cmd, inc = 60):
    # enter用来安排某事件的发生时间，从现在起第n秒开始启动
    schedule.enter(inc, 0, perform_command, (cmd, inc))
    # 持续运行，直到计划时间队列变成空为止
    schedule.run()


print("show time after 10 seconds:")
timming_exe("echo %time%", 10)

