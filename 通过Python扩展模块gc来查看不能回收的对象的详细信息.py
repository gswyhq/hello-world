#!/usr/bin/python
# -*- coding:utf8 -*- #
from __future__ import  generators
from __future__ import  division
from __future__ import  print_function
from __future__ import  unicode_literals
import sys,os,json

if sys.version >= '3':
    PY3 = True
else:
    PY3 = False

if PY3:
    import pickle
else:
    import cPickle as pickle
    from codecs import open


#--------------- code begin --------------
# -*- coding: utf-8 -*-
import gc
import sys
 
def memory_garbage(f):
    """内存泄露测试"""
    def wrap(*args, **kwargs):
        # 启用垃圾自动回收.
        gc.enable()
        # 启用垃圾收集调试标识.
        if PY3:
            gc.set_debug(gc.DEBUG_COLLECTABLE | gc.DEBUG_UNCOLLECTABLE | gc.DEBUG_STATS | gc.DEBUG_LEAK )
        else:
            gc.set_debug(gc.DEBUG_COLLECTABLE | gc.DEBUG_UNCOLLECTABLE | \
                    gc.DEBUG_INSTANCES | gc.DEBUG_OBJECTS)

        print ('泄露测试...')
        ret = f(*args, **kwargs)
        print ('开始收集...')
        _unreachable = gc.collect()
        print ('不可达到对象编号:%d' % _unreachable)
        print ('垃圾对象编号:%d' % len(gc.garbage))
        print('发现的不可达（即是垃圾对象）、但又不能释放（即不能回收）的对象:',gc.garbage)
        # gc.garbage 是一个 list 对象，列表项是垃圾收集器发现的不可达（即是垃圾对象）、但又不能释放（即不能回收）的对象。
        return ret
    return wrap
    
class CGcLeak(object):
    def __init__(self):
        self._text = '#'*10

    def __del__(self):
        pass
 
def make_circle_ref1():
    _gcleak = CGcLeak()
    _gcleak._self = _gcleak # 让 _gcleak 形成一个自己对自己的循环引用。
    # 对自己的循环引用，多个对象间的循环引用都会导致内存泄漏。
    print ('变量_gcleak 的引用计数:%d' % sys.getrefcount(_gcleak))
    del _gcleak
    try:
        print ('_gcleak ref count1:%d' % sys.getrefcount(_gcleak))
    except UnboundLocalError:
        print ('_gcleak 变为不可达(unreachable)的非法变量.')


class CGcLeakA(object):
    def __init__(self):
        self._text = '#'*10

    def __del__(self):
        pass
 
class CGcLeakB(object):
    def __init__(self):
        self._text = '*'*10

    def __del__(self):
        pass
 
@memory_garbage
def make_circle_ref2():
    _a = CGcLeakA()
    _b = CGcLeakB()
    _a._b = _b # test_code_2
    _b._a = _a # test_code_3
    print ('引用计数:a=%d b=%d' % (sys.getrefcount(_a), sys.getrefcount(_b)))
    #_b._a = None  # test_code_4
    del _a
    del _b
    try:
        print ('引用计数1:a=%d' % sys.getrefcount(_a))
    except UnboundLocalError:
        print ('_a 变为不可达(unreachable)的非法变量!')
    try:
        print ('引用计数2:b=%d' % sys.getrefcount(_b))
    except UnboundLocalError:
        print ('_b is 变为不可达(unreachable)的非法变量!')
        
if __name__ == '__main__':
    #test_gcleak()
    make_circle_ref2()
    
    
