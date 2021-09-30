#!/usr/bin/python3
# coding: utf-8

# Pool.map()多参数任务

# 在给map方法传入带多个参数的方法不能达到预期的效果，像下面这样

def job(x ,y):
    return x * y

def text1():
    pool = multiprocessing.Pool()
    res = pool.map(job, 2, 3)
    print(res)

# 所以只能通过对有多个参数的方法进行封装，在进程中运行封装后的方法如下

def job33(x ,y):
    return x * y


def job1(z):
    return job33(z[0], z[1])


def test2():
    pool = multiprocessing.Pool()
    res = pool.map(job1, [(2, 3), (3, 4)])
    print(res)


# 这样就能达到传递多个参数的效果
# ps：如果需要得到多个结果可以传入多个元组在一个列表中

# Pool.apply_async()输出多个迭代结果

# 在使用apply_async()方法接收多个参数的方法时，在任务方法中正常定义多个参数，参数以元组形式传入即可
# 但是给apply_async()方法传入多个值获取多个迭代结果时就会报错，因为该方法只能接收一个值，所以可以将该方法放入一个列表生成式中，如下

def job4(x):
    return x * x

def test3():
    pool = multiprocessing.Pool()
    res = [pool.apply_async(job4, (i,)) for i in range(3)]
    print( [r.get() for r in res])

# python 3中提供了starmap和startmap_async两个方法




if __name__ == '__main__':
    main()