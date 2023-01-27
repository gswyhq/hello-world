
# 我们正常dump一次 所以load一次就好了
# 若dump了两次 则要load两次才能把数据读出来

# pickle模块是python中用来将Python对象序列化和解序列化的一个工具。“pickling”是将Python对象转化为字节流的过程，而“unpickling”是相反的过程（将来自“binary file或bytes-like object”的字节流反转为对象的过程）。
#
# 5种协议
# Protocol version 0 是最原始一种协议，它向后与以前的Python版本兼容。
# Protocol version 1 是一种老的二进制格式，它也兼容以前版本的Python。
# Protocol version 2 是在 Python2.3 中被引入的。它提供了对新类型new-style class更加高效的pickling。
# Protocol version 3 是在 Python3.0 中加入的。它明确的支持了字节对象bytes objects的pickling，且不能被Python2.x unpickling。这个协议就是默认的协议，也是在Python3的其他版本中推荐的协议。
# Protocol version 4 是在Python3.4中被加入的。它增加了对超大对象的支持，且支持pickling的对象类型也更多。

import os
import pickle
data1 = {'a': [1, 2.0, 3, 4+6j],
         'b': ('string', u'Unicode string'),
         'c': None}
data2 = {'aa': [1, 2.0, 3, 4+6j],
         'bb': ('string', u'Unicode string'),
         'cc': None}

pkfile=open("testfile.txt",'ab')
pickle.dump(data1,pkfile)
pickle.dump(data2,pkfile)
pkfile.close()

pkfile2=open("testfile.txt",'rb')
pkf=pickle.load(pkfile2)
pkf1=pickle.load(pkfile2)

print(pkf)
print(pkf1)

# Pickle 每次序列化生成的字符串有独立头尾，pickle.load() 只会读取一个完整的结果，所以你只需要在 load 一次之后再 load 一次，就能读到第二次序列化的结果。
# 如果不知道文件里有多少 pickle 对象，可以在 while 循环中反复 load 文件对象，直到抛出异常为止。


# pickle.HIGHEST_PROTOCOL
# 这是一个整数，表示最高可用的协议版本。这个值可以作为参数protocol传给函数dump()和dumps()以及Pickler构造器。
# pickle.DEFAULT_PROTOCOL
# 这是一个整数，表示用来pickling的默认协议版本。可能比pickle.HIGHEST_PROTOCOL小。目前默认的协议版本是3，协议3是专门为Python3设计的一种新的协议。

# pickle 序列化对象报错：
# AttributeError: Can't pickle local object 'get_lookup_encoding.<locals>.<lambda>'
# 这个时候可以使用 dill 来代替：
import dill
fun_a = lambda x: x+3
with open('fun_a.pkl', 'wb') as f:
    dill.dump(fun_a, f)
with open('fun_a.pkl', 'rb') as f:
    fun_b = dill.load(f)
fun_b(10)

# pickle 分片，分块存储：
import pickle, math
import numpy as np
filename = "myfile.pkl"
batch_size = 32
data = np.random.random(size=(1000,))
with open(filename, 'wb') as file_handle:
    for item in range(math.ceil(data.shape[0]/batch_size)):
        pickle.dump(data[item*batch_size:item*batch_size+batch_size], file_handle)

with open(filename, 'rb') as file_handle:
    try:
        while True:
            result = pickle.load(file_handle)
            print(result)
    except EOFError:
        print('读取完成！')

