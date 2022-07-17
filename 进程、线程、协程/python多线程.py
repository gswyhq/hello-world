#! /usr/lib/python3
# -*- coding: utf-8 -*-
#http://www.cnblogs.com/kaituorensheng/p/4445418.html

import threading
import time,os
import queue

def worker(keywords,keywords_list,lock):
    #keywords=set()
    while 1:
        keyword=keywords_list.get()
        #print('ok',keyword)
        #print(os.getpid())
        print('添加前1：',keyword)
        with lock:
            print('添加前：',keywords)
            keywords.add(time.ctime()[14:20]+keyword)

            time.sleep(1)
            if len(keywords)>3:
                print('添加后：',keywords)
                keywords=set()#若将keywords设置为外部变量，或者类变量则，此句在这里根本不起作用
        keywords_list.task_done()
    #程序压根就执行不到此处来
    print('添加后2：',keywords)


def main():
    keywords=set()
    in_list=[i for i in 'zhwiki.chn.json']
    keywords_list = queue.Queue()  # 构造一个FIFO队列先进先出。以获得代理的ip和端口 ip:ports
    lock = threading.Lock()  # 创建锁 good_proxies, bad_proxies lists
    for i in range(5):
        p = threading.Thread(name='线程'+str(1+i),target = worker, args = (keywords,keywords_list,lock))
    #p.daemon = True #若设置daemon为真，则父进程终止后，即使子进程没有完成，也被强制终止。
        print('当前线程：',p.name)
        p.setDaemon(1)
        p.start()
    for line in in_list:
        keywords_list.put(line)
    keywords_list.join()# 主线程会停在这里，直到所有数字被get()，并且task_done()，因为没有调用task_done()，所在这里会一直阻塞，直到用户按^C

    print('完成。')

if __name__=='__main__':
    main()


'''
类变量在多线程下是共享的，还有一个就是没意识到 内存释放问题，导致越累越大

1.python 类变量 在多线程情况 下的 是共享的

2.python 类变量 在多线程情况 下的 释放是不完全的

3.python 类变量 在多线程情况 下没释放的那部分 内存 是可以重复利用的
公用的数据，除非是只读的，不然不要当类成员变量，一是会共享，二是不好释放。

'''
