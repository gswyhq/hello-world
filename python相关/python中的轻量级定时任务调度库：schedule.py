#!/usr/bin/python3
# coding: utf-8

import schedule
import time

def job():
    print("I'm working...")

schedule.every(10).minutes.do(job)
schedule.every().hour.do(job)
schedule.every().day.at("10:30").do(job)
schedule.every().monday.do(job)
schedule.every().wednesday.at("13:15").do(job)

def main():
    while True:
        schedule.run_pending()
        time.sleep(1)

'''
schedule其实就只是个定时器。在while True死循环中，schedule.run_pending()是保持schedule一直运行，去查询上面那一堆的任务，在任务中，就可以设置不同的时间去运行。跟crontab是类似的。
 
但是，如果是多个任务运行的话，实际上它们是按照顺序从上往下挨个执行的。如果上面的任务比较复杂，会影响到下面任务的运行时间。

'''
if __name__ == '__main__':
    main()