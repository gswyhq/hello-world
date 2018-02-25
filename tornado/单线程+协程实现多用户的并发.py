#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Michael Liao'

'''
async web application.
'''

# 来源： https://www.liaoxuefeng.com/wiki/0014316089557264a6b348958f449949df42a6d3a2e542c000/0014320981492785ba33cc96c524223b2ea4e444077708d000

import asyncio

from aiohttp import web
import time

# 使用 async def foo() 在函数定义的时候代替 @gen.coroutine 装饰器, 用 await 代替yield.
# 如果把asyncio.sleep(1)换成time.sleep(1)，将会有这样的报错 > TypeError: object NoneType can't be used in 'await' expression （报错了3次）
# 为什么？因为time.sleep()返回的是None，它不async所以报错了，await等的是一个协程对象。而asyncio.sleep()则很async，返回了协程对象。await终于等到了他要的对象。

# 如果使用time.sleep因为是单线程会阻塞整个应用

# 协程有以下几个特性：
# 1. 使用了`async`关键字的函数，虽然还是函数，返回值不再是None，而是一个协程实例coroutine。
# 2. 和3.4的语法对比，@asyncio.coroutine替换为async；yield from替换为await。
# 3. 一个异步函数并不能直接执行，直接执行都会引起`RuntimeWarning: coroutine '××××' was never awaited`,必须在EventLoop中执行。

async def index(request):
    print('start1, ', time.time())
    await asyncio.sleep(1.5)  # 通过await关键字通知ioloop循环，我要异步等待这个函数的调用。

    print('end1, ', time.time())
    return web.Response(body=b'<h1>Index</h1>')

async def hello(request):

    print('start2, ', time.time())
    # await asyncio.sleep(1.5)
    time.sleep(1)
    text = '<h1>hello, %s!</h1>' % request.match_info['name']

    print('end2, ', time.time())
    return web.Response(body=text.encode('utf-8'))

async def init(loop):
    app = web.Application(loop=loop)
    app.router.add_route('GET', '/', index)
    app.router.add_route('GET', '/hello/{name}', hello)
    srv = await loop.create_server(app.make_handler(), '127.0.0.1', 8000)
    print('Server started at http://127.0.0.1:8000...')
    return srv

loop = asyncio.get_event_loop()
loop.run_until_complete(init(loop))
loop.run_forever()
