#!/usr/bin/python3

import time
from apistar import ASyncApp, Route

async def hello_world() -> dict:
    # We can perform some network I/O here, asyncronously.
    return {'hello': 'async', 'time': time.time()}

async def homepage() -> str:
    time.sleep(10) # 异步函数里头同步的调用，会阻塞整个进程; 可使用asyncio.run_in_executor，将同步变为异步。
    return '<html><body><h1>Homepage</h1></body></html>'

async def welcome(name=None):
    """
    开机问候语
    :param name: 参数
    :return: 字典
    """
    if name is None:
        return {'message': 'Welcome to API Star!'}
    return {'message': 'Welcome to API Star, %s!' % name}


routes = [
    Route('/hello', method='GET', handler=hello_world),
    Route('/', method='GET', handler=welcome),
    Route('/homepage', method='GET', handler=homepage),
]

app = ASyncApp(routes=routes)

def main():
    app.serve('0.0.0.0', 5000, use_debugger=False, use_reloader=False)

if __name__ == '__main__':
    main()

# 请求http://localhost:5000/homepage时， 会阻塞http://localhost:5000/hello的请求
#  async关键词定义了个协程函数，Python 中的协程是运行在一条线程中通过消息队列调控的，如果运行的线程堵塞了那么就会造成消息队列阻塞。

