#!/usr/bin/python3
# coding: utf-8


class asd(object):

    def __new__(cls, *args, **kwargs):
        r = super(asd,cls).__new__(cls)
        r.initialize(*args)  # 只要继承了asd类，就会将initialize方法作为初始化方法
        return r

    def initialize(self, name=''):
        print('你说对方')

class bnm(asd):

    def initialize(self, name=''):
        print('bnm_initialize is running')

class foo(asd):

    def initialize(self, name=''):
        self.name = name
        print('foo_initialize is running, my name is %s' %(self.name))


r = bnm()
r1 = foo('linghuchong')
