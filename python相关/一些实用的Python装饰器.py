#/usr/lib/python3.5
# -*- coding: utf-8 -*-

#函数执行耗时统计
import time
def timing(f):
    def wrap(*args):
        print ('<function name: {0}>'.format(f.func_name))
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        print ('[timecosts: {0} ms]'.format((time2-time1)*1000.0))
        return ret
    return wrap

#使用方法：
@timing
def hello(name):
    print ("hello, %s" % name)
 
hello('tom')


#给函数调用做缓存
from functools import wraps
def memo(fn):
    cache = {}
    miss = object()
    @wraps(fn)
    def wrapper(*args):
        result = cache.get(args, miss)
        if resultis miss:
            result = fn(*args)
            cache[args] = result
        return result
    return wrapper
#使用方法：
@memo
def fib(n):
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)
 
fib(100) #这里其实可以使用上面的函数耗时统计的decorator进行耗时比对
fib(101) #用以判断memo装饰器是否起到作用


#失败重试函数
import time
import math
 
# Retry decorator with exponential backoff
def retry(tries, delay=3, backoff=2):
    '''Retries a function or method until it returns True.
    delay sets the initial delay in seconds, and backoff sets the factor by which
    the delay should lengthen after each failure. backoff must be greater than 1,
    or else it isn't really a backoff. triesmustbeatleast 0, and delay
    greaterthan 0.'''
    if backoff <= 1:
        raiseValueError("backoff must be greater than 1")
    tries = math.floor(tries)
    if tries < 0:
        raiseValueError("tries must be 0 or greater")
    if delay <= 0:
        raiseValueError("delay must be greater than 0")
    def deco_retry(f):
        def f_retry(*args, **kwargs):
            mtries, mdelay = tries, delay # make mutable
            rv = f(*args, **kwargs) # first attempt
            while mtries > 0:
                if rvis True: # Done on success
                    return True
                mtries -= 1      # consume an attempt
                time.sleep(mdelay) # wait...
                mdelay *= backoff  # make future wait longer
                rv = f(*args, **kwargs) # Try again
            return False # Ran out of tries :-(
        return f_retry # true decorator -> decorated function
    return deco_retry  # @retry(arg[, ...]) -> true decorator
#使用方法：
@retry(5)
def some_func():
    pass
 
some_func()


#超时退出函数
#这个函数的作用在于可以给任意可能会hang住的函数添加超时功能，这个功能在编写外部API调用 、网络爬虫、数据库查询的时候特别有用
import signal, functools
class TimeoutError(Exception): pass #定义一个Exception，在超时情况下抛出
 
def timeout(seconds, error_message = 'Function call timed out'):
    def decorated(func):
        def _handle_timeout(signum, frame):
            raiseTimeoutError(error_message)
        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result
        return functools.wraps(func)(wrapper)
    return decorated
#使用方法：
@timeout(5) #限定下面的slowfunc函数如果在5s内不返回就强制抛TimeoutError Exception结束
def slowfunc(sleep_time):
    import time
    time.sleep(sleep_time) #这个函数就是休眠sleep_time秒
 
slowfunc(3)    #sleep 3秒，正常返回，没有异常
slowfunc(10)  #被终止



#打印调试信息函数
import sys, os, linecache
def trace(f):
    def globaltrace(frame, why, arg):
        if why == "call": return localtrace
        return None
    def localtrace(frame, why, arg):
        if why == "line":
            filename = frame.f_code.co_filename
            lineno = frame.f_lineno
            bname = os.path.basename(filename)
            print "{}({}): {}".format(bname, lineno, linecache.getline(filename, lineno)),
        return localtrace
    def _f(*args, **kwds):
        sys.settrace(globaltrace)
        result = f(*args, **kwds)
        sys.settrace(None)
        return result
    return _f
#使用方法：
@trace
def simple_print():
    print (1)
    print (22)
    print (333)
 
simple_print() #调用

# 带参数的装饰器
In[39]: def decorator_factory(enter_message, exit_message):
    # We're going to return this decorator
    def simple_decorator(f):
        def wrapper(key):
            print enter_message
            f(key)
            print exit_message
 
        return wrapper
 
    return simple_decorator
In[40]: @decorator_factory("Start", "End")
def hello(key):
    print "Hello World:{}".format(key)
In[41]: hello('234')
Start
Hello World:234
End


def main():
    pass


if __name__ == "__main__":
    main()
