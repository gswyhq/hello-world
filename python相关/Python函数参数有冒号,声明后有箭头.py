#!/usr/bin/python3
# coding=utf-8

def f(ham: str, eggs: str = 'eggs') -> str :
    print("Annotations:", f.__annotations__)
    print("Arguments:", ham, eggs)
    return [ham , ' and ' , eggs]

print(f("test","abc"))
print(f("test",123456))
print(f("test"))

# 写一个:str 和 ->str是什么意思呢？
#
# 在官方文档指明.__annotations__是函数的参数注释和返回值注释：
#
# 所以打印出Annotations: {'ham': <class 'str'>, 'eggs': <class 'str'>, 'return': <class 'str'>}
#
# 其实并没有指定类型 只是写函数的人提醒用函数的人最好传什么类型的参数，因为最后需要两个参数进行字符串拼接;
#
# 当然，也可以直接写字符串提醒:

def f2(ham: "传一个字符串", eggs: str = 'eggs') -> str :
    print("Annotations:", f.__annotations__)
    print("Arguments:", ham, eggs)
    return ham + ' and ' + eggs

print(f("test",123))
# 而声明函数后那个箭头："->" 是返回值的注释，-> str 意思即是提醒函数使用者返回值会是一个str型。

# 需要注意的是：
# 调用函数传入值的类型若不是冒号后面定义的类型，函数也能运行；
# 函数返回值，若不是箭头后面定义的类型，函数也能运行。

def main():
    pass


if __name__ == '__main__':
    main()