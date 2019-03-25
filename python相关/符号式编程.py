#!/usr/bin/python3
# coding=utf-8

# 符号式编程的程序需要的三个步骤：
# 1 定义计算流程
# 2 把计算流程编译成可执行的程序；
# 3 给定输入，调用编译好的程序执行。

def add_str():
    return """
def add(a, b):
    return a + b    
"""

def fancy_func_str():
    return """
def fancy_func(a, b ,c ,d):
    e = add(a, b)
    f = add(c, d)
    g = add(e, f)
    return g
"""

def evoke_str():
    return add_str() + fancy_func_str() + """
print(fancy_func(1, 2, 3, 4))    
"""

prog = evoke_str()
print(prog)

# 通过compile函数编译完整的计算流程并运行
y = compile(prog, "", "exec")
exec(y)

def main():
    pass


if __name__ == '__main__':
    main()