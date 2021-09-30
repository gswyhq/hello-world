import time
import asyncio
from asyncio import coroutine, events, futures

def for_test():
    for i in range(3):
        yield i
print(list(for_test()))

def yield_from_test():
    yield from range(3)
print(list(yield_from_test()))

async def others():
    print("start")
    await asyncio.sleep(2)
    print('end')
    return '返回值'

async def func():
    print("执行协程函数内部代码")
    # await等待对象的值得到结果之后再继续向下走
    response = await others()
    print("IO请求结束，结果为：", response)

# asyncio.run( func() )
loop = asyncio.get_event_loop()
loop.run_until_complete( func() )

def time_sleep(seconds, future=None):
    # print('sleep: {}s'.format(seconds))
    time.sleep(seconds)
    result = '{}秒'.format(seconds)
    if future:
        return futures._set_result_unless_cancelled(future, result)
    else:
        return result


@coroutine
def sleep(seconds=1, loop=None):
    if loop is None:
        loop = asyncio.get_event_loop()
        # loop = asyncio.new_event_loop()  # 若创建新的event_loop 会报错： got Future <Future pending> attached to a different loop
    future = loop.create_future()  # 创建一个附加到事件循环的asyncio.Future对象。
    h = future._loop.call_later(0, time_sleep, seconds, future)
    try:
        return (yield from future) # 这一句，有两个动作第一个是把未完成future传递给事件循环，让事件循环继续运行；第二个等到任务真正完成的时候，事件循环到了才发送结果到 return的右边的临时变量，再返回。
    finally:
        h.cancel()

async def async_sleep(seconds=1, loop=None):
    # return await sleep(seconds=seconds, loop=loop)  # 若不通过线程池的话，虽说是将同步任务转换了异步任务，但实际上还是同步执行。4个time.sleep(2)的任务，耗时为8s;
    return await loop.run_in_executor(None, time_sleep, seconds)  # 通过线程池，将同步任务转换为异步任务；4个time.sleep(2)的任务，耗时约为2s.

@asyncio.coroutine
def func1(loop=None):
    print(1)
    # yield from asyncio.sleep(2)  # 遇到IO耗时操作，自动化切换到tasks中的其他任务
    yield from async_sleep(2, loop=loop)
    print(2)

@asyncio.coroutine
def func2(loop=None):
    print(3)
    # yield from asyncio.sleep(2) # 遇到IO耗时操作，自动化切换到tasks中的其他任务
    yield from async_sleep(2, loop=loop)
    print(4)


async def func3(loop=None):
    print(5)
    await async_sleep(2, loop=loop)
    print(6)

async def func4(loop=None):
    print(7)
    await async_sleep(2, loop=loop)
    print(8)

loop = asyncio.get_event_loop()

tasks = [
    asyncio.ensure_future( func1(loop=loop), loop=loop ),
    asyncio.ensure_future( func2(loop=loop), loop=loop ),
    asyncio.ensure_future( func3(loop=loop), loop=loop ),
    asyncio.ensure_future( func4(loop=loop), loop=loop ),

    # 通过asyncio.create_task(协程对象)的方式创建Task对象，这样可以让协程加入事件循环中等待被调度执行。
    # asyncio.get_event_loop().create_task(func1(loop=loop)),
    # asyncio.get_event_loop().create_task(func2(loop=loop)),
    # asyncio.get_event_loop().create_task(func3(loop=loop)),
    # asyncio.get_event_loop().create_task(func4(loop=loop))
]

start_time = time.time()
loop.run_until_complete(asyncio.wait(tasks, loop=loop))
print(time.time() - start_time)
# 若不通过线程池的话，虽说是将同步任务转换了异步任务，但实际上还是同步执行。4个time.sleep(2)的任务，耗时为8s;
# 通过线程池，将同步任务转换为异步任务；4个time.sleep(2)的任务，耗时约为2s.
