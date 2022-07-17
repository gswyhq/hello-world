
from fastapi import APIRouter
import time
import asyncio
from asyncio import events, futures, coroutine

router = APIRouter()

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
        loop = events.get_event_loop()
    future = loop.create_future()  # 创建一个附加到事件循环的asyncio.Future对象。
    h = future._loop.call_later(0, time_sleep, seconds, future)
    try:
        return (yield from future) # 这一句，有两个动作第一个是把未完成future传递给事件循环，让事件循环继续运行；第二个等到任务真正完成的时候，事件循环到了才发送结果到 return的右边的临时变量，再返回。
    finally:
        h.cancel()

async def async_sleep(seconds=1, loop=None):
    return await sleep(seconds=seconds, loop=loop)

async def sleep4():
    loop = asyncio.get_event_loop()
    tasks = [loop.create_task(sleep(3, loop)), loop.create_task(sleep(1, loop)), loop.create_task(sleep(2, loop))]  # 创建任务, 不立即执行
    res = await asyncio.wait(tasks)  # 执行并等待所有任务完成
    return res

@router.get("/a")
async def a():
    time.sleep(1)  # 异步函数里头同步的调用，会阻塞整个进程
    return {"message": "异步模式，但是同步执行sleep函数，执行过程是串行的"}


@router.get("/b")
async def b():
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, time.sleep, 1)
    return {"message": "线程池中运行sleep函数"}


@router.get("/c")
async def c():
    await asyncio.sleep(1)
    return {"message": "异步模式，且异步执行sleep函数"}


@router.get("/d")
def d():
    time.sleep(1)
    return {"message": "同步模式，但是FastAPI会放在线程池中运行，所以很快"}

@router.get("/e")
async def e():
    loop = asyncio.get_event_loop()         # 创建事件循环
    loop.call_soon(time_sleep, 1)
    # call_soon 将普通函数当作 task 加入到事件循环并排定执行顺序
    # 该方法的第一个参数为普通函数名字，普通函数的参数写在后面
    return {"message": "异步模式，call_soon排定执行，不关心返回值"}

@router.get("/f")
async def f():
    ret = await sleep()
    # print(ret)
    return {"message": "异步模式，阻塞io通过future保存未来的结果"}

@router.get("/g")
async def g():
    # t1 = time.time()
    loop = asyncio.get_event_loop()
    future = loop.create_future()  # 创建一个附加到事件循环的asyncio.Future对象。
    await loop.run_in_executor(None, time_sleep, 1, future)
    ret1 = future.result()  # 获取返回结果
    # print(ret1)

    return {"message": "异步模式，通过线程池将阻塞io转换为异步io"}

def app02():
    """
    多个异步任务并行执行，耗时约为最大耗时的任务
    :return:
    """
    async def fn1():
        await asyncio.sleep(3)
        print("fn1 ...")
    async def fn2():
        await asyncio.sleep(2)
        print("fn2 ...")
    async def fn3():
        await asyncio.sleep(5)
        print("fn3 ...")
    loop = asyncio.get_event_loop()
    tasks = [
        fn1(),
        fn2(),
        fn3()
    ]
    loop.run_until_complete(asyncio.wait(tasks))
    loop.close()

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(router, host="0.0.0.0", port=8000)

# 压测每个的耗时(在60秒内发请求，一次100个请求)：
# docker run -it --rm devth/alpine-bench -t 60 -c 100 http://10.113.78.56:8000/a
# 16761.755 [ms
# docker run -it --rm devth/alpine-bench -t 60 -c 100 http://10.113.78.56:8000/b
# 50.924 [ms]
# docker run -it --rm devth/alpine-bench -t 60 -c 100 http://10.113.78.56:8000/c
# 13.401 [ms
# docker run -it --rm devth/alpine-bench -t 60 -c 100 http://10.113.78.56:8000/d
# 50.935 [ms

# 压测每个的耗时（-n共发出200个请求，-c模拟100并发，相当100人同时访问）
# docker run -it --rm devth/alpine-bench -n 200 -c 100 http://10.113.78.56:8000/a
# 1002.812 [ms， 总耗时：100281.233 ms
# docker run -it --rm devth/alpine-bench -n 200 -c 100 http://10.113.78.56:8000/b
# 50.701 [ms， 总耗时：5070.074 ms
# docker run -it --rm devth/alpine-bench -n 200 -c 100 http://10.113.78.56:8000/c
# 12.488 [ms，总耗时：1248.763 ms
# docker run -it --rm devth/alpine-bench -n 200 -c 100 http://10.113.78.56:8000/d
# 50.879 [ms， 总耗时：5087.855 ms
# docker run -it --rm devth/alpine-bench -n 200 -c 100 http://10.113.78.56:8000/e
# 1004.637 [ms，总耗时：100463.719 ms
# docker run -it --rm devth/alpine-bench -n 200 -c 100 http://10.113.78.56:8000/g
# 50.862 [ms，总耗时：5086.214 ms

# import time, requests
# for url in ['http://10.113.78.56:8000/'+i for i in list('abcd')]:
#     start_time = time.time()
#     for _ in range(100):
#         r = requests.get(url)
#         r.json().keys()
#     print("{} 百次请求耗时: {}s".format(url, time.time()-start_time))
#
# http://10.113.78.56:8000/a 百次请求耗时: 100.61593270301819s
# http://10.113.78.56:8000/b 百次请求耗时: 100.71041321754456s
# http://10.113.78.56:8000/c 百次请求耗时: 100.24041795730591s
# http://10.113.78.56:8000/d 百次请求耗时: 100.78421330451965s

# /a接口：
# fastapi框架会将async函数会放到event loop（事件轮询）中运行。
# 虽然使用了async，但是函数内部并没有用到await，所以堵塞了。
# 相当于单线程运行。执行过程是串行的，所以总耗时100秒。

# /b接口：
# 利用asyncio异步IO获取当前的event loop。
# 然后将time.sleep(1)放到一个event loop中去运行，函数内部用到了await，所以无堵塞。
# 执行过程是并行的，但受到线程池的限制，所以总耗时5秒。

# /c接口：
# 使用异步IO的sleep取代了普通的同步sleep。
# 原理与/b接口一致，但不受线程池限制。
# 执行过程是并行的，所以总耗时1秒。

# /d接口：
# 这个函数没有async修饰，即一个普通函数。
# 但是FastAPI会将函数放到thread pool中执行。
# 服务器是4核CPU，线程池的默认配置是核心数4*5=20。
# 服务器在第一秒和第二秒分别处理20个请求，第三秒处理20个请求...
# 所以100个并发总耗时5秒。

# /e接口：
# 虽然使用了async，但call_soon遵循FIFO原则（先入先出）,还是会block的（运算block）, 比如耗时的操作time.sleep(1).
# 效果上等同于 /a接口（使用了async，内部是阻塞的），

# /f接口：
# 等同于接口f （唯一不同就是有返回值）

# /g接口：
# 本质上还是线程池运行阻塞io, 等同于接口/b，/d

# 总结：
# 官方说，无论你是否使用async，FastAPI都会采用异步的方式处理。
# 但是，如果你定义了async函数，函数体却是同步的调用（例：/a接口），将导致函数执行过程变成串行。
# 所以，若不能保证是异步调用，就直接定义成普通函数（/d接口）即可，效果与异步+线程池（/b接口）是一样的，但效果上肯定差于异步函数异步调用（/c接口）。

# 任务： 1.IO密集型（会有cpu空闲的时间）  注：time.sleep等同于IO操作， socket通信也是IO
#         2.计算密集型
# 　　time.sleep是IO密集型的例子，由于涉及到了加法和乘法，属于计算密集型操作(串行可能比多线程还快)，那么，就产生了一个结论，多线程对于IO密集型任务有作用，而计算密集型任务不推荐使用多线程。
# 由于GIL锁，多线程不可能真正实现并行，所谓的并行也只是宏观上并行微观上并发，本质上是由于遇到io操作不断的cpu切换所造成并行的现象。由于cpu切换速度极快，所以看起来就像是在同时执行。
# 对于计算密集型采用多线程问题：没有利用多核的优势，这就造成了多线程不能同时执行，并且增加了切换的开销，串行的效率可能更高。
# 　　同步与异步是对应的，它们是线程之间的关系，两个线程之间要么是同步的，要么是异步的。
# 　　阻塞与非阻塞是对同一个线程来说的，在某个时刻，线程要么处于阻塞，要么处于非阻塞。
# 多核计算密集型的,选择多进程,可以充分利用cpu
# 多核IO密集型的,选择多线程,线程开销小
