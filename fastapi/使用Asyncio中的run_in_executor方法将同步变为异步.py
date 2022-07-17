
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor

def a(abcd):
    time.sleep(1)
    return abcd
    

async def c():
    # loop = asyncio.get_running_loop() # python3.7+
    loop = asyncio.get_event_loop() # python3.6
    return await loop.run_in_executor(None, a, 'abc123')

thread_executor = ThreadPoolExecutor(5)
async def d():
    future = thread_executor.submit(a, '123456')
    # 上面的thread_executor.submit返回的是一个future对象，但是并不是一个符合asyncio模块的future是不可等待的，即无法调用await去等待该对象。
    # 通过wrap_future函数可以将concurrent.futures.Future变成asyncio.Future实现可等待。
    return await asyncio.wrap_future(future)

def test1():
    # asyncio.run(c()) # python3.7+
    ret = asyncio.get_event_loop().run_until_complete(c()) # python3.6
    print(ret)

def test2():
    ret = asyncio.get_event_loop().run_until_complete(d()) # python3.6
    print(ret)

if __name__ == '__main__':
    test1()
    test2()
