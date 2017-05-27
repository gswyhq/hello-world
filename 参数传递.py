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


def main():
    pass


if __name__ == "__main__":
    main()
