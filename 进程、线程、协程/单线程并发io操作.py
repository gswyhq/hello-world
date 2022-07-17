#!/usr/bin/python3
# coding: utf-8

import asyncio
import time


def wait_tiem():
    time.sleep(1)

async def compute(x, y):
    print("Compute %s + %s ..." % (x, y))
    await asyncio.sleep(1.0)
    # wait_tiem()
    return x + y


async def compute2(x, y):
    print("Compute %s + %s ..." % (x, y))
    await asyncio.sleep(1.0)
    # wait_tiem()
    return x + y + 4

async def print_sum(x, y):
    result1 = await compute(x, y)
    result2 = await compute2(x, y)
    print("%s + %s = %s" % (x, y, result1))
    return {"ret1": result1, "ret2": result2}

# async和await是针对coroutine的新语法，要使用新的语法，只需要做两步简单的替换：

# 把@asyncio.coroutine替换为async；
# 把yield from替换为await。


def main():
    start_tiem = time.time()

    # loop = asyncio.get_event_loop()
    # 第二次使用loop的时候程序就会抛出异常RuntimeError: Event loop is closed,这也无可厚非，理想的程序也应该是在一个时间循环中解决掉各种异步IO的问题。
    # 但放在终端环境如Ipython中，如果想要练习Python的异步程序的编写的话每次都要重新开启终端未免太过于麻烦，这时候要探寻有没有更好的解决方案。
    # 解决方案:我们可以使用asyncio.new_event_loop函数建立一个新的事件循环
    loop = asyncio.new_event_loop()
    # t = loop.run_until_complete(print_sum(1, 2))
    tasks = [compute(1, 3), compute2(2, 4)]
    t = loop.run_until_complete(asyncio.wait(tasks))
    loop.close()

    print(t)
    end_time = time.time()
    print(end_time - start_tiem)


if __name__ == '__main__':
    main()