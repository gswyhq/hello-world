#!/usr/bin/python3
# coding: utf-8

from concurrent.futures import ThreadPoolExecutor
from functools import partial, wraps
import time

import tornado.ioloop
import tornado.web


executor = ThreadPoolExecutor(max_workers=2)
# 对象的submit 和 map方法可以用来启动线程池中线程执行任务
# 当线程数超过线程池总数时，就会等待

def f(a,b):
    print(a,b, time.time())
    time.sleep(1)
    return a ** b

future = executor.submit(f,2,3)

data = future.result()

print(data)
executor.map(f,[2,3,4],[4,5,6])


import concurrent.futures
import time
def sleeper(n):
    time.sleep(n)
    print(time.time())
    return time.time()

def test1():
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        print(list(executor.map(sleeper, [1,1])))
        data = executor.submit(sleeper, 1)

        print(data.result())



from concurrent.futures import ProcessPoolExecutor


def test2():
    with ProcessPoolExecutor(max_workers=3) as executor:
        print(list(executor.map(sleeper, [1, 1])))
        data = executor.submit(sleeper, 1)

        print(data.result())

def main():
    pass


if __name__ == '__main__':
    main()
