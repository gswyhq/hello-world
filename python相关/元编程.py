#/usr/lib/python3.5
# -*- coding: utf-8 -*-
from __future__ import  generators
from __future__ import  division
from __future__ import  print_function
from __future__ import  unicode_literals
import sys,os,json

if sys.version >= '3':
    PY3 = True
else:
    PY3 = False

# type有一种完全不同的能力，它也能动态的创建类。type可以接受一个类的描述作为参数，然后返回一个类。
#type(类名, 父类的元组（针对继承的情况，可以为空），包含属性的字典（名称和值）)

#type 接受一个字典来为类定义属性，因此
class Foo(object):
    bar = True
#可以翻译为：
Foo = type(str('Foo'), (), {'bar':True})

class N:
    def __init__(self,base):
        self.base = base

    def __getattr__(self,name):
        def add(*arg):
            s = self.base
            for i in arg:
                s = s + i
            return s
        def mul(*arg):
            s = self.base
            for i in arg:
                s = s*i
            return s
        if name == 'add':
            return add
        else:
            return mul
            
b = N(2)
#print b.add1(1,2,3)
print (b.xyz(3,4))
print (b.add(1,2))


# 请记住，'type'实际上是一个类，就像'str'和'int'一样
# 所以，你可以从type继承
class UpperAttrMetaClass(type):
    # __new__ 是在__init__之前被调用的特殊方法
    # __new__是用来创建对象并返回之的方法
    # 而__init__只是用来将传入的参数初始化给对象
    # 你很少用到__new__，除非你希望能够控制对象的创建
    # 这里，创建的对象是类，我们希望能够自定义它，所以我们这里改写__new__
    # 如果你希望的话，你也可以在__init__中做些事情
    # 还有一些高级的用法会涉及到改写__call__特殊方法，但是我们这里不用
    def __new__(upperattr_metaclass, future_class_name, future_class_parents, future_class_attr):
        '''返回一个类对象，将属性都转为大写形式'''
        attrs = ((name, value) for name, value in future_class_attr.items() if not name.startswith('__'))
        uppercase_attr = dict((name.upper(), value) for name, value in attrs)
        return type(future_class_name, future_class_parents, uppercase_attr)

# 这种方式其实不是OOP。我们直接调用了type，而且我们没有改写父类的__new__方法。
class UpperAttrMetaclass(type):
    def __new__(upperattr_metaclass, future_class_name, future_class_parents, future_class_attr):
        '''返回一个类对象，将属性都转为大写形式'''
        attrs = ((name, value) for name, value in future_class_attr.items() if not name.startswith('__'))
        uppercase_attr = dict((name.upper(), value) for name, value in attrs)
 
        # 复用type.__new__方法
        # 这就是基本的OOP编程，没什么魔法
        return type.__new__(upperattr_metaclass, future_class_name, future_class_parents, uppercase_attr)
        
# 元类会自动将你通常传给‘type’的参数作为自己的参数传入
def upper_attr(future_class_name, future_class_parents, future_class_attr):
    '''返回一个类对象，将属性都转为大写形式'''
    #  选择所有不以'__'开头的属性
    attrs = ((name, value) for name, value in future_class_attr.items() if not name.startswith('__'))
        # 将它们转为大写形式
    uppercase_attr = dict((name.upper(), value) for name, value in attrs)
 
    # 通过'type'来做类对象的创建
    return type(future_class_name, future_class_parents, uppercase_attr)

if PY3:
    class Foo(metaclass=upper_attr):
        bar = 'bip'
else:
    class Foo(object):
        # 我们也可以只在这里定义__metaclass__，这样就只会作用于这个类中
        __metaclass__ = upper_attr  #  这会作用到这个模块中的所有类
        bar = 'bip'
    
print (hasattr(Foo, 'bar'))
# 输出: False
print (hasattr(Foo, 'BAR'))
# 输出:True
 
f = Foo()
print (f.BAR)
# 输出:'bip'

def f(name, bases, attrs):
    attrs['c'] = 2
    return type(name, bases, attrs) #type并不是一个函数，而是构造了一个类

A = f(str('A'), (), {'b': 1})
a = A()
print (A, a.b, a.c)

# 以上代码等价于：
if PY3:
    class A(metaclass=f):
        b = 1
else:
    class A(object):
        __metaclass__ = f 
        #__metaclass__实际上，就是指创建类A的时候，要用什么函数进行生成。，这是python2中的写法。
        b = 1

a = A()
print (A, a.b, a.c)

#为什么舍弃function，而使用元类。function固然简单，但是function是无法继承的。这里不仅仅指我们无法创建一个Meta的子类，扩充meta的行为。而且，使用function的类，一旦继承，其子类是不会管父类的__metaclass__定义的。


