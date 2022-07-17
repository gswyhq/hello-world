#!/usr/bin/python3
# coding: utf-8

import time
import multiprocessing

from multiprocessing import Pool
from multiprocessing.dummy import Pool

def wait_tiem():
    time.sleep(1)

def compute(x, y):
    print("Compute %s + %s ..." % (x, y))
    # await asyncio.sleep(1.0)
    wait_tiem()
    return x + y


def compute2(x, y):
    print("Compute %s + %s ..." % (x, y))
    # await asyncio.sleep(1.0)
    wait_tiem()
    return x + y + 4

def main():
    start_tiem = time.time()

    # 下面进程池代码属于父进程代码，若放在IDE上运行，可能会报错：
    # AttributeError: Can't get attribute 'func' on <module '__main__' (built-in)>
    # pool = multiprocessing.Pool(processes=4)  # 进程池
    with multiprocessing.Pool(processes=4) as pool: # 进程池
#    with multiprocessing.dummy.Pool(processes=4) as pool:  # 线程池
        jobs = []
        p = pool.apply_async(func=compute, args=(1, 2))
        # apply_async(func[, args[, kwds[, callback]]]) 它是非阻塞，apply(func[, args[, kwds]])是阻塞的
    
        jobs.append(p)
    
        p = pool.apply_async(func=compute2, args=(11, 12))
        jobs.append(p)
        # pool.close() # 关闭pool，使其不在接受新的任务。调用join之前，先调用close函数，否则会出错。执行完close后不会有新的进程加入到pool,join函数等待所有子进程结束；
        # pool.join()
        # terminate()    结束工作进程，不在处理未完成的任务。
        # join()    主进程阻塞，等待子进程的退出， join方法要在close或terminate之后使用。
        
        for ret in jobs:
            print(ret.get())

    end_time = time.time()
    print("总耗时：", end_time - start_tiem)


if __name__ == '__main__':
    main()
