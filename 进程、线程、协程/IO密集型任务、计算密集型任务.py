#coding=utf-8
import sys
import multiprocessing
import time
import threading
import math
import traceback

from multiprocessing import Pool
from multiprocessing.dummy import Pool

g_search_list = list(range(10000))

# 定义一个IO密集型任务：利用time.sleep()
def io_task(n, data=None):
    time.sleep(1)
    return {'n': n, "data": data}

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
        jobs = [pool.apply_async(func=io_task, args=(i, ), kwds={"data": "data_{}".format(i)}) for i in range(test_count)]
        pool.close()
        pool.join()
        for ret in jobs:
            result = ret.get()
            print("结果：", result)
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

# python multiprocessing.Pool 中map、map_async、apply、apply_async的区别
#              多参数Multi-args   并发Concurrence    阻塞Blocking     有序结果Ordered-results
# map          no           yes            yes          yes
# apply        yes          no             yes          no
# map_async    no           yes            no           yes
# apply_async  yes          yes            no           no

import time
import traceback
import multiprocessing.dummy
def my_callback(d):
    print(f'执行结果：{d}')
    return d

def my_error_callback(e):
    print("执行出现错误：{}，错误详情：{}".format(e, traceback.format_exc()))
    return e

def my_func(a, b, c=None):
    if a>0:
        time.sleep(a)
    print((a, b, c))
    return a+b+c

def ordered_results():
    result_list = []
    data_list = [(1, 2, 3),
                 (2, 3, 4),
                 (-1, 2, 3),
                 (4, 6, 9)]
    with multiprocessing.dummy.Pool(processes=20) as pool:  # 线程池
        jobs = [pool.apply_async(func=my_func, args=(a, b),
                                 kwds={"c": c},
                                 callback=my_callback, error_callback=my_error_callback) for a, b, c in data_list]
        # print("批量请求数：", len(data_list))
        pool.close()
        pool.join()
        for ret in jobs:
            result = ret.get(timeout=2)
            result_list.append(result)
    print('最终的结果是有顺序的是：', result_list)
    return result_list

start_time = time.time()
ordered_results()
print('总耗时：', time.time()-start_time)
# (-1, 2, 3)
# 执行结果：4
# (1, 2, 3)
# 执行结果：6
# (2, 3, 4)
# 执行结果：9
# (4, 6, 9)
# 执行结果：19
# 最终的结果是有顺序的是： [6, 9, 4, 19]
# 总耗时： 4.015351057052612



