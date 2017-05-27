# -*- coding: utf-8 -*-
"""
Created on Sun Sep  4 22:32:51 2016

Python 之 @property
"""

#定义类Student，拥有变量名name和score
class Student(object):
  def __init__(self,name,score):
      self.name = name
      self.score = score
#这样，我们可以在类外修改Student的实例的成员变量:
s1 = Student()
s1.name = "Lily"
s1.score = 9999 # 这里的定义是不合理的

#但是，上述这样定义score是不会进行参数检查的，也就意味着我们不能执行必要的参数以及错误处理。
#我们可以定义相应的set和get成员函数来访问成员变量score，并且进行参数检查。如下所示：
class Student(object):
    def __init__(self,name,score):
        self.name = name
        self.score = score
    def get_score(self): return self.score
    def set_score(self,score):
        if not isinstance(score, int):
            raise ValueError("invalid score!!!")
        if score < 0 or score > 100:
            raise ValueError("score must be between [0,100]!!!")
        self._score = score
#上述代码定义了score成员的set和get函数。（可能实际应用时，修改分数比较常见）
#现在，我们改变参数的代码是这样的：
s1 = Student()
s1.set_score(9999) #这里会抛出异常
#上述的第二种方式实现了set函数的参数检查，但是修改score的代码从简单的 s1.score = 90 变成了 s1.set_score(90) .我们怎么样才能做到既检验输入的参数又使得修改score的代码不变呢？

#@Property便是这个作用。
#下面，我们讨论Python的高级特性 @Property。简单的说@Properyty就是将成员函数的调用变成属性赋值。

class Student(object):
    def __init__(self,name,score):
        self._name = name
        self._score = score
    @property # @property的用处，将函数调用转化为属性访问
    def score(self): return self._score
    @score.setter # @score.setter 便是针对与 score函数包裹的成员变量的的set函数。当我们需要修改_score的值时，使用score函数，但是就像score是类的成员属性一样
    def score(self,score):
        if not isinstance(score,int):
            raise ValueError("invalid score!!!")
        if score < 0 or score > 100:
            raise ValueError("score must be between [0,100]!!!")
        self._score = score
    @property
    def name(self): return self._name
s1 = Student("Lily", 90)
#s1.name = "Luly"
s1.score = 100

# 尽量不要让函数名与变量名同名


