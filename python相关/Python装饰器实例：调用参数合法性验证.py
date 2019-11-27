#!/usr/bin/python
# -*- coding:utf8 -*- #
from __future__ import generators
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import sys

if sys.version >= '3':
    PY3 = True
else:
    PY3 = False

if PY3:
    import pickle
else:
    import cPickle as pickle
    from codecs import open






def type_limit(*typeLimit,**returnType):
    def test_value_type(func):
        def wrapper(*param,**kw):
            length = len(typeLimit)
            if length != len(param):
                print("类型限制和参数列表必须具有相同的长度")
                raise TypeError
            for index in range(length):
                t = typeLimit[index]
                p = param[index]
                if not isinstance(p,t):
                    print("参数 %s 类型应该是： %s,而不是： %s !"%(str(p),type(t()),type(p)))
                    raise TypeError #如果引发Error异常，后面的代码将不能执行
            res = func(*param,**kw)
            if "returnType" in returnType:
                limit = returnType["returnType"]
                if  not isinstance(res,limit):
                    print("这个函数的返回值类型应该是： %s,而不是： %s"%(limit,type(res) ) )
                    raise TypeError
            return res
        return wrapper
    return test_value_type


@type_limit(int,int,returnType=int)
def test(x,y):
    return x+y

test(2,2)
#以上代码等价于：
#temp = type_limit(int,int) #temp =  test_value_type
#test = temp(test) #这是，test已经在原test上经过修饰，指向wrapper


##################################################################################################################
def argfilter(*types):
    def deco(func):
    #这是修饰器
        def newfunc(*args):
            #新的函数
            if len(types)==len(args):
                correct = True
                for i in range(len(args)):
                    if not isinstance(args[i], types[i]):
                #判断类型
                        correct = False
                if correct:
                    return func(*args)
                #返回原函数值
                else:
                    raise TypeError
            else:
                raise TypeError
        return newfunc
        #由修饰器返回新的函数
    return deco
        #返回作为修饰器的函数

@argfilter(int, bool)
#指定参数类型
def func(i, s):
#定义被修饰的函数
    print (i, s)

func(25,True)
#之后你想限制类型的话, 就这样:
 #@argfilter(第一个参数的类名, 第二个参数的类名, ..., 第N个参数的类名)
 #def yourfunc(第一个参数, 第一个参数, ..., 第N个参数):
 #

 #
 #相当于:
 #def yourfunc(第一个参数, 第一个参数, ..., 第N个参数):
 #

 #yourfunc = argfilter(第一个参数的类名, 第二个参数的类名, ..., 第N个参数的类名)(yourfunc)



'''
@summary: 验证器
该模块提供了一个装饰器用于验证参数是否合法，使用方法为：

from validator import validParam, nullOk, multiType

@validParam(i=int)
def foo(i):
    return i+1

编写验证器：

1. 仅验证类型：
@validParam(type, ...)
例如：
检查第一个位置的参数是否为int类型：
@validParam(int)
检查名为x的参数是否为int类型：
@validParam(x=int)

验证多个参数：
@validParam(int, int)
指定参数名验证：
@validParam(int, s=str)

针对*和**参数编写的验证器将验证这些参数实际包含的每个元素：
@validParam(varargs=int)
def foo(*varargs): pass

@validParam(kws=int)
def foo7(s, **kws): pass

2. 带有条件的验证：
@validParam((type, condition), ...)
其中，condition是一个表达式字符串，使用x引用待验证的对象；
根据bool(表达式的值)判断是否通过验证，若计算表达式时抛出异常，视为失败。
例如：
验证一个10到20之间的整数：
@validParam(i=(int, '10<x<20'))
验证一个长度小于20的字符串：
@validParam(s=(str, 'len(x)<20'))
验证一个年龄小于20的学生：
@validParam(stu=(Student, 'x.age<20'))

另外，如果类型是字符串，condition还可以使用斜杠开头和结尾表示正则表达式匹配。
验证一个由数字组成的字符串：
@validParam(s=(str, '/^\d*$/'))

3. 以上验证方式默认为当值是None时验证失败。如果None是合法的参数，可以使用nullOk()。
nullOk()接受一个验证条件作为参数。
例如：
@validParam(i=nullOk(int))
@validParam(i=nullOk((int, '10<x<20')))
也可以简写为：
@validParam(i=nullOk(int, '10<x<20'))

4. 如果参数有多个合法的类型，可以使用multiType()。
multiType()可接受多个参数，每个参数都是一个验证条件。
例如：
@validParam(s=multiType(int, str))
@validParam(s=multiType((int, 'x>20'), nullOk(str, '/^\d+$/')))

5. 如果有更复杂的验证需求，还可以编写一个函数作为验证函数传入。
这个函数接收待验证的对象作为参数，根据bool(返回值)判断是否通过验证，抛出异常视为失败。
例如：
def validFunction(x):
    return isinstance(x, int) and x>0
@validParam(i=validFunction)
def foo(i): pass

这个验证函数等价于：
@validParam(i=(int, 'x>0'))
def foo(i): pass


@author: HUXI
@since: 2011-3-22
@change:
'''
"""
import inspect
import re

class ValidateException(Exception): pass


def validParam(*varargs, **keywords):
    '''验证参数的装饰器。'''

    varargs = map(_toStardardCondition, varargs)
    keywords = dict((k, _toStardardCondition(keywords[k]))
                    for k in keywords)

    def generator(func):
        args, varargname, kwname = inspect.getargspec(func)[:3]
        dctValidator = _getcallargs(args, varargname, kwname,
                                    varargs, keywords)

        def wrapper(*callvarargs, **callkeywords):
            dctCallArgs = _getcallargs(args, varargname, kwname,
                                       callvarargs, callkeywords)

            k, item = None, None
            try:
                for k in dctValidator:
                    if k == varargname:
                        for item in dctCallArgs[k]:
                            assert dctValidator[k](item)
                    elif k == kwname:
                        for item in dctCallArgs[k].values():
                            assert dctValidator[k](item)
                    else:
                        item = dctCallArgs[k]
                        assert dctValidator[k](item)
            except:
                raise ValidateException, ('%s() parameter validation fails, param: %s, value: %s(%s)'% (func.func_name, k, item, item.__class__.__name__))

            return func(*callvarargs, **callkeywords)

        wrapper = _wrapps(wrapper, func)
        return wrapper

    return generator


def _toStardardCondition(condition):
    '''将各种格式的检查条件转换为检查函数'''

    if inspect.isclass(condition):
        return lambda x: isinstance(x, condition)

    if isinstance(condition, (tuple, list)):
        cls, condition = condition[:2]
        if condition is None:
            return _toStardardCondition(cls)

        if cls in (str, unicode) and condition[0] == condition[-1] == '/':
            return lambda x: (isinstance(x, cls)
                              and re.match(condition[1:-1], x) is not None)

        return lambda x: isinstance(x, cls) and eval(condition)

    return condition


def nullOk(cls, condition=None):
    '''这个函数指定的检查条件可以接受None值'''

    return lambda x: x is None or _toStardardCondition((cls, condition))(x)


def multiType(*conditions):
    '''这个函数指定的检查条件只需要有一个通过'''

    lstValidator = map(_toStardardCondition, conditions)
    def validate(x):
        for v in lstValidator:
            if v(x):
                return True
    return validate


def _getcallargs(args, varargname, kwname, varargs, keywords):
    '''获取调用时的各参数名-值的字典'''

    dctArgs = {}
    varargs = tuple(varargs)
    keywords = dict(keywords)

    argcount = len(args)
    varcount = len(varargs)
    callvarargs = None

    if argcount <= varcount:
        for n, argname in enumerate(args):
            dctArgs[argname] = varargs[n]

        callvarargs = varargs[-(varcount-argcount):]

    else:
        for n, var in enumerate(varargs):
            dctArgs[args[n]] = var

        for argname in args[-(argcount-varcount):]:
            if argname in keywords:
                dctArgs[argname] = keywords.pop(argname)

        callvarargs = ()

    if varargname is not None:
        dctArgs[varargname] = callvarargs

    if kwname is not None:
        dctArgs[kwname] = keywords

    dctArgs.update(keywords)
    return dctArgs


def _wrapps(wrapper, wrapped):
    '''复制元数据'''

    for attr in ('__module__', '__name__', '__doc__'):
        setattr(wrapper, attr, getattr(wrapped, attr))
    for attr in ('__dict__',):
        getattr(wrapper, attr).update(getattr(wrapped, attr, {}))

    return wrapper


#===============================================================================
# 测试
#===============================================================================


def _unittest(func, *cases):
    for case in cases:
        _functest(func, *case)


def _functest(func, isCkPass, *args, **kws):
    if isCkPass:
        func(*args, **kws)
    else:
        try:
            func(*args, **kws)
            assert False
        except ValidateException:
            pass

def _test1_simple():
    #检查第一个位置的参数是否为int类型：
    @validParam(int)
    def foo1(i): pass
    _unittest(foo1,
              (True, 1),
              (False, 's'),
              (False, None))

    #检查名为x的参数是否为int类型：
    @validParam(x=int)
    def foo2(s, x): pass
    _unittest(foo2,
              (True, 1, 2),
              (False, 's', 's'))

    #验证多个参数：
    @validParam(int, int)
    def foo3(s, x): pass
    _unittest(foo3,
              (True, 1, 2),
              (False, 's', 2))

    #指定参数名验证：
    @validParam(int, s=str)
    def foo4(i, s): pass
    _unittest(foo4,
              (True, 1, 'a'),
              (False, 's', 1))

    #针对*和**参数编写的验证器将验证这些参数包含的每个元素：
    @validParam(varargs=int)
    def foo5(*varargs): pass
    _unittest(foo5,
              (True, 1, 2, 3, 4, 5),
              (False, 'a', 1))

    @validParam(kws=int)
    def foo6(**kws): pass
    _functest(foo6, True, a=1, b=2)
    _functest(foo6, False, a='a', b=2)

    @validParam(kws=int)
    def foo7(s, **kws): pass
    _functest(foo7, True, s='a', a=1, b=2)


def _test2_condition():
    #验证一个10到20之间的整数：
    @validParam(i=(int, '10<x<20'))
    def foo1(x, i): pass
    _unittest(foo1,
              (True, 1, 11),
              (False, 1, 'a'),
              (False, 1, 1))

    #验证一个长度小于20的字符串：
    @validParam(s=(str, 'len(x)<20'))
    def foo2(a, s): pass
    _unittest(foo2,
              (True, 1, 'a'),
              (False, 1, 1),
              (False, 1, 'a'*20))

    #验证一个年龄小于20的学生：
    class Student(object):
        def __init__(self, age): self.age=age

    @validParam(stu=(Student, 'x.age<20'))
    def foo3(stu): pass
    _unittest(foo3,
              (True, Student(18)),
              (False, 1),
              (False, Student(20)))

    #验证一个由数字组成的字符串：
    @validParam(s=(str, r'/^\d*$/'))
    def foo4(s): pass
    _unittest(foo4,
              (True, '1234'),
              (False, 1),
              (False, 'a1234'))


def _test3_nullok():
    @validParam(i=nullOk(int))
    def foo1(i): pass
    _unittest(foo1,
              (True, 1),
              (False, 'a'),
              (True, None))

    @validParam(i=nullOk(int, '10<x<20'))
    def foo2(i): pass
    _unittest(foo2,
              (True, 11),
              (False, 'a'),
              (True, None),
              (False, 1))


def _test4_multitype():
    @validParam(s=multiType(int, str))
    def foo1(s): pass
    _unittest(foo1,
              (True, 1),
              (True, 'a'),
              (False, None),
              (False, 1.1))

    @validParam(s=multiType((int, 'x>20'), nullOk(str, '/^\d+$/')))
    def foo2(s): pass
    _unittest(foo2,
              (False, 1),
              (False, 'a'),
              (True, None),
              (False, 1.1),
              (True, 21),
              (True, '21'))

def _main():
    d = globals()
    from types import FunctionType
    print()
    for f in d:
        if f.startswith('_test'):
            f = d[f]
            if isinstance(f, FunctionType):
                f()

if __name__ == '__main__':
    _main()

"""
