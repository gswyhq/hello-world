#coding=utf-8
import sys
import multiprocessing
import time
import threading
import math

from multiprocessing import Pool
from multiprocessing.dummy import Pool

g_search_list = list(range(10000))

# 定义一个IO密集型任务：利用time.sleep()
def io_task(n):
    time.sleep(1)
    return n

# 定义一个计算密集型任务：利用一些复杂加减乘除、列表查找等
def cpu_task(count = 0):
    for i in range(10000):
        count += pow(3 * 2, 3 * 2)+math.pow(i+1, 0.5)+math.sin(i) if i in g_search_list else 0
    return count

if __name__ == '__main__':
    print("cpu count:", multiprocessing.cpu_count(), "\n")
    # thread_count = multiprocessing.cpu_count()
    thread_count = 20
    process_count = multiprocessing.cpu_count()
    test_count = 200
    print(u"========== 直接执行IO密集型任务 ==========")
    time_0 = time.time()
    for i in range(test_count):
        ret = io_task(i)
        # print(ret)
    print(u"结束：", time.time() - time_0, "\n")

    print("========== 多线程执行IO密集型任务 ==========")
    time_0 = time.time()

    with multiprocessing.dummy.Pool(processes=thread_count) as pool:  # 线程池
        jobs = [pool.apply_async(func=io_task, args=(i, )) for i in range(test_count)]
        for ret in jobs:
            result = ret.get()
            # print(result)
    print("结束：", time.time() - time_0, "\n")
    print("========== 多进程执行IO密集型任务 ==========")

    time_0 = time.time()
    # 下面进程池代码属于父进程代码，若放在IDE上运行，可能会报错：
    # AttributeError: Can't get attribute 'func' on <module '__main__' (built-in)>
    with multiprocessing.Pool(processes=process_count) as pool:  # 进程池
        jobs = [pool.apply_async(func=io_task, args=(i,)) for i in range(test_count)]
        for ret in jobs:
            result = ret.get()
            # print(result)
    print("结束：", time.time() - time_0, "\n")
    print("========== 直接执行CPU密集型任务 ==========")

    time_0 = time.time()
    for i in range(test_count):
        ret = cpu_task(i)
        # print(ret)
    print(u"结束：", time.time() - time_0, "\n")

    print("========== 多线程执行CPU密集型任务 ==========")
    time_0 = time.time()

    with multiprocessing.dummy.Pool(processes=thread_count) as pool:  # 线程池
        jobs = [pool.apply_async(func=cpu_task, args=(i,)) for i in range(test_count)]
        for ret in jobs:
            result = ret.get()
            # print(result)
    print("结束：", time.time() - time_0, "\n")

    print("========== 多进程执行cpu密集型任务 ==========")
    time_0 = time.time()
    # 下面进程池代码属于父进程代码，若放在IDE上运行，可能会报错：
    # AttributeError: Can't get attribute 'func' on <module '__main__' (built-in)>
    with multiprocessing.Pool(processes=process_count) as pool:  # 进程池
        jobs = [pool.apply_async(func=cpu_task, args=(i,)) for i in range(test_count)]
        for ret in jobs:
            result = ret.get()
            # print(result)
    print("结束：", time.time() - time_0, "\n")

# cpu count: 4
#
# ========== 直接执行IO密集型任务 ==========
# 结束： 200.11495161056519
#
# ========== 多线程执行IO密集型任务 ==========
# 结束： 50.10620379447937
#
# ========== 多进程执行IO密集型任务 ==========
# 结束： 50.82318115234375
#
# ========== 直接执行CPU密集型任务 ==========
# 结束： 129.39605736732483
#
# ========== 多线程执行CPU密集型任务 ==========
# 结束： 126.53654503822327
#
# ========== 多进程执行cpu密集型任务 ==========
# 结束： 65.0292227268219

# ========== 20个多线程执行IO密集型任务 ==========
# 结束： 10.116163492202759

# ========== 20个多线程执行CPU密集型任务 ==========
# 结束： 132.36442589759827 
