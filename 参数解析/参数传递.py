#/usr/lib/python3.5
# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import unicode_literals

def fun_var_args(farg, *args):  
    print ("arg:", farg  )
    for value in args:  
        print ("another arg:", value  )
        
    print(args[0])
    print(args[1])
  
fun_var_args(1, "two", 3) # *args可以当作可容纳多个变量组成的list  




def fun_var_kwargs(farg, **kwargs):  
    print ("arg:", farg  )
    for key in kwargs:  
        print ("another keyword arg: %s: %s" % (key, kwargs[key])  )
  
  
fun_var_kwargs(farg=1, myarg2="two", myarg3=3) # myarg2和myarg3被视为key， 感觉**kwargs可以当作容纳多个key和value的dictionary 









def fun_var_args_list(arg1, arg2, arg3):  
    print ("arg1:", arg1  )
    print ("arg2:", arg2  )
    print ("arg3:", arg3  )
  
args = ["two", 3] #list  
  
fun_var_args_list(1, *args)  



def fun_var_args_dict(arg1, arg2, arg3):  
    print ("arg1:", arg1  )
    print ("arg2:", arg2  )
    print ("arg3:", arg3  )
  
kwargs = {"arg3": 3, "arg2": "two"} # dictionary  
  
fun_var_args_dict(1, **kwargs)  


#*args与位置参数和默认参数混用:*args要放到位置参数后面，默认参数要放最后。
def foo(x,*args,y = 1):
    print(x)
    print(y)
    print(args)
foo(1,2,3,4,5,6,7,8,9,10,y=10000)   #调用函数
# 或者（调用时关键词参数不能有关键字）：
def func(name, age, sex=1, *args, **kargs):
    print name, age, sex, args, kargs
func('tanggu', 25, 2, 'music', 'sport', class=2)
# tanggu 25 1 ('music', 'sport') {'class'=2}

# 同时使用*args和**kwargs时，必须*args参数列要在**kwargs前

# 混合参数混合使用：
# 示例1：
def foo(x,*args,a=4,**kwargs):　　#使用默认参数时，注意默认参数的位置要在args之后kwargs之前
    print(x)
    print(a)
    print(args)
    print(kwargs)

foo(1,5,6,7,8,y=2,z=3)  #调用函数，不修改默认参数
#1   #x的值
#4   #a的值
#(5, 6, 7, 8)   #*args的值
#{'y': 2, 'z': 3}    ##kwargs的值

# 示例2：
def foo(x,a=4,*args,**kwargs):　　##注意：当需要修改默认参数时，要调整默认参数的位置，要放在args之前即可，但不可放在开头。
    print(x)
    print(a)
    print(args)
    print(kwargs)

foo(1,9,5,6,7,8,y=2,z=3)    #调用函数，修改默认参数a为9
#1   #x的值
#9   #被修改后a的值
#(5, 6, 7, 8)    #args的值
#{'y': 2, 'z': 3}    #kwargs的值

# 示例3：
def foo(x,*args,a=4,**kwargs):  #使用默认参数时，注意默认参数的位置要在args之后kwargs之前
    print(x)
    print(a)
    print(args)
    print(kwargs)

foo(1,*(5,6,7,8),**{"y":2,"z":3})   #调用函数，不改默认参数
#1   #x的值
#4   #默认a的值
#(5, 6, 7, 8)    #args的值
#{'y': 2, 'z': 3}    #kwargs的值
#-------------分割线----------------------
def foo(x,a=4,*args,**kwargs):  #注意：当需要修改默认参数时，要调整默认参数的位置，要放在args之前，但不可放在开头
    print(x)
    print(a)
    print(args)
    print(kwargs)

foo(1,9,10,*(5,6,7,8),**{"y":2,"z":3})  #调用函数，修改默认参数a为9
#1   #x的值
#9   #修改默认参数a后的值
#(10, 5, 6, 7, 8)    #args的值
#{'y': 2, 'z': 3}    #kwargs的值
#
#位置参数：
#
#调用函数时所传参数的位置必须与定义函数时参数的位置相同
#
#关键字参数：
#
#使用关键字参数会指定参数值赋给哪个形参，调用时所传参数的位置可以任意
#
#*位置参数：可接受任意数量的位置参数(元组)；只能作为最后一个位置参数出现，其后参数均为关键字参数
#
#**关键字参数：可接受任意数量的关键字参数(字典)；只能作为最后一个参数出现
#
# 
#
#默认参数：默认参数的赋值只会在函数定义的时候绑定一次，默认值不会再被修改

def main():
    pass


if __name__ == "__main__":
    main()
