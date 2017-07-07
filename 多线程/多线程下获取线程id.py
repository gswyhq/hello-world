#!/usr/bin/python3
# coding: utf-8

import os
import threading
import ctypes
import time
import requests

def pthread_level1(i):
    print ("任务号 :%s"%i)
    #获取threading对象的标识ident
    print (threading.currentThread())
    print (threading.currentThread().ident)
    print ("当前的线程 id: ",ctypes.CDLL('libc.so.6').syscall(186))
    d = requests.get("http://www.google.com")
    time.sleep(10)
    return

def main():
    l = []
    for i in range(5):
        t = threading.Thread(target=pthread_level1,args=(i,))
        l.append(t)
    for i in l:
        i.start()
    #查看进程跟线程的关系, 调用shell命令，查看某个进程下的线程
    os.system("pstree -p " + str(os.getpid()))
    for i in l:
        i.join()
    print ("Sub-process done.")

if __name__ == '__main__':
    main()