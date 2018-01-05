#!/usr/bin/python3
# coding: utf-8

from threading import Thread
from time import sleep

'''
定义了一个装饰器 async 和 A 、B 两个function 
A 里面sleep 20s , 然后打印 a function 字符串 
B 里面直接打印 b function 字符串 
我们顺序调用两个功能： 
A(） 
B( ) 
实际结果： 
b function 
20s… 
a function
'''

def async(f):
    def wrapper(*args, **kwargs):
        thr = Thread(target = f, args = args, kwargs = kwargs)
        thr.start()
    return wrapper

@async
def A():
    sleep(20)
    print ("a function")
    return 'A'

def B():
    print ("b function")
    return 'B'


def main():
    A()
    return B()


if __name__ == '__main__':
    ret = main()
    print(ret)