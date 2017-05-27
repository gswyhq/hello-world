#/usr/lib/python3.5
# -*- coding: utf-8 -*-

#https://segmentfault.com/a/1190000000414339

import time
from pprint import pprint
from multiprocessing.dummy import Pool as ThreadPool 

def print_file(f,m=time.time()):
    t=m%1
    time.sleep(t)
    print(t,'*'*10,f)
    #return (m,'*'*10,f)

def main():
    urls = [
        'http://www.python.org', 
        'http://www.python.org/about/',
        'http://www.onlamp.com/pub/a/python/2003/04/17/metaclasses.html',
        'http://www.python.org/doc/',
        'http://www.python.org/download/',
        'http://www.python.org/getit/',
        'http://www.python.org/community/',
        'https://wiki.python.org/moin/',
        'http://planet.python.org/',
        'https://wiki.python.org/moin/LocalUserGroups',
        'http://www.python.org/psf/',
        'http://docs.python.org/devguide/',
        'http://www.python.org/community/awards/'
        # etc.. 
        ]

    # 实例化 Pool 对象：
    # Pool 对象有一些参数，这里我所需要关注的只是它的第一个参数：processes.
    # 这一参数用于设定线程池中的线程数。其默认值为当前机器 CPU 的核数。
    # 一般来说，执行 CPU 密集型任务时，调用越多的核速度就越快。
    # 但是当处理网络密集型任务时，事情有有些难以预计了，通过实验来确定线程池的大小才是明智的。
    # 可以为 CPU 密集型任务和 IO 密集型任务分别选择多进程和多线程库来进一步提高执行速度
    with ThreadPool(4)  as pool:
        results = pool.map(print_file, urls)
        #若需要传递多参数，需使用starmap
        #results = pool.starmap(print_file, zip(urls,[time.time()]*len(urls)))
        #pprint("{}{}".format(results,time.time()))
    
if __name__=="__main__":
    start=time.time()
    main()
    print('运行时间：',time.time()-start)

