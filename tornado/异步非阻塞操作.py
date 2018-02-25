#!/usr/bin/python3
# coding: utf-8

import asyncio
import time
import aiohttp
from tornado import gen
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from tornado.httpclient import AsyncHTTPClient

http_client = AsyncHTTPClient()
# response = await http_client.fetch(url)

thread_pool = ThreadPoolExecutor(4)

@gen.coroutine
def call_blocking():
    yield thread_pool.submit(sleep_1)


async def sleep_1():
    time.sleep(1)
    return time.time()

async def slow_operation(n):
    await asyncio.sleep(1)
    ret = await sleep_1()
    # ret = await download(1)

    print("Slow operation {} complete".format(n))
    print(ret)

async def main():
    await asyncio.wait([
        slow_operation(1),
        slow_operation(2),
        slow_operation(3),
    ])



#!/usr/bin/env python
# encoding:utf-8
import asyncio
import aiohttp
import time


async def download(url): # 通过async def定义的函数是原生的协程对象
    print("get: %s" % url)
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            print(url, resp.status)
            # response = await resp.read()
    # 创建了一个 ClientSession 对象命名为session，然后通过session的get方法得到一个 ClientResponse 对象，命名为resp，
    # get方法中传入了一个必须的参数url，就是要获得源码的http url。至此便通过协程完成了一个异步IO的get请求。

# 仅仅是把涉及I/O操作的代码封装到async当中是不能实现异步执行的。必须使用支持异步操作的非阻塞代码才能实现真正的异步。目前支持非阻塞异步I/O的库是aiohttp


async def main2():
    start = time.time()
    await asyncio.wait([
        download("http://www.163.com"),
        download("http://www.mi.com"),
        download("http://www.baidu.com"),
        download('http://127.0.0.1:8000/hello/123'),
        download('http://127.0.0.1:8000/')
    ])
    end = time.time()
    print("Complete in {} seconds".format(end - start))





if __name__ == '__main__':
    start_time = time.time()
    loop = asyncio.get_event_loop()
    # loop.run_until_complete(main())
    loop.run_until_complete(main2())

    print(time.time()-start_time)