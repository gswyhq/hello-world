

# NumPy提供了多种存取数组内容的文件操作函数。保存数组数据的文件可以是二进制格式或者文本格式。二进制格式的文件又分为NumPy专用的格式化二进制类型和无格式类型。
# 一，tofile()和fromfile()
# tofile()将数组中的数据以二进制格式写进文件
# tofile()输出的数据不保存数组形状和元素类型等信息
# fromfile()函数读回数据时需要用户指定元素类型，并对数组的形状进行适当的修改

import os
import numpy as np
USERNAME = os.getenv("USERNAME")

# 一，tofile()和fromfile()
# tofile()将数组中的数据以二进制格式写进文件
# tofile()输出的数据不保存数组形状和元素类型等信息
# fromfile()函数读回数据时需要用户指定元素类型，并对数组的形状进行适当的修改

t = np.random.randint(0, 5, size=(2,3))
t
Out[15]:
array([[3, 3, 2],
       [1, 3, 4]])

t.tofile(rf"D:\Users\{USERNAME}\Downloads\test\abc.bin")
b = np.fromfile(rf"D:\Users\{USERNAME}\Downloads\test\abc.bin")
b.shape
Out[26]: (3,)
b
Out[27]: array([6.36598737e-314, 2.12199579e-314, 8.48798317e-314])
b = np.fromfile(rf"D:\Users\{USERNAME}\Downloads\test\abc.bin", dtype='int32')
b
Out[35]: array([3, 3, 2, 1, 3, 4])
b.shape = (2, 3)
b
Out[37]:
array([[3, 3, 2],
       [1, 3, 4]])

从上面的例子可以看出，在读入数据时:需要正确设置dtype参数，并修改数组的shape属性才能得到和原始数据一致的结果。
无论数据的排列顺序是C语言格式还是Fortran语言格式，tofile()都统一使用C语言格式输出。
此外如果指定了sep参数，则fromfile()和tofile()将以文本格式对数组进行输入输出。sep参数指定的是文本数据中数值的分隔符。
embeddings = np.random.random(size=(10, 32))
embeddings.tofile(rf"D:\Users\{USERNAME}\Downloads\test\abc.csv", sep=', ', format='%.8f')
embeddings2 = np.fromfile(rf"D:\Users\{USERNAME}\Downloads\test\abc.csv", sep=', ')
embeddings2.shape = 10, 32

二.save()和load()
NumPy专用的二进制格式保存数据，它们会自动处理元素类型和形状等信息
如果想将多个数组保存到一个文件中，可以使用savez()
save()的第一个参数是文件名，其后的参数都是需要保存的数组，也可以使用关键字参数为数组起名
非关键字参数传递的数组会自动起名为arr_0、arr_1、…
savez()输出的是一个扩展名为npz的压缩文件，其中每个文件都是一个save()保存的npy文件，文件名和数组名相同
load()自动识别npz文件，并且返回一个类似于字典的对象，可以通过数组名作为键获取数组的内容

a = np.random.randint(1, 5, size=(2, 3))
a
Out[50]:
array([[4, 3, 2],
       [3, 1, 3]])
np.save(rf"D:\Users\{USERNAME}\Downloads\test\a.npy", a)
np.load(rf"D:\Users\{USERNAME}\Downloads\test\a.npy")
Out[52]:
array([[4, 3, 2],
       [3, 1, 3]])

b = np.array((1, 0.2, 0.3))
c = np.sin(b)
np.savez(rf"D:\Users\{USERNAME}\Downloads\test\abc.npz", a, b, c=c)
abc = np.load(rf"D:\Users\{USERNAME}\Downloads\test\abc.npz")

# 获取非关键字参数值：
abc['arr_0']
Out[62]:
array([[4, 3, 2],
       [3, 1, 3]])
abc['arr_1']
Out[63]: array([1. , 0.2, 0.3])

# 获取关键字参数值：
abc['c']
Out[64]: array([0.84147098, 0.19866933, 0.29552021])

三.savetxt()和loadtxt()
读写1维和2维数组的文本文件
可以用它们读写CSV格式的文本文件
np.savetxt(rf"D:\Users\{USERNAME}\Downloads\test\abc.txt", embeddings)  # 缺省安装'%.18e'格式保存数值，以空格分隔
np.loadtxt(rf"D:\Users\{USERNAME}\Downloads\test\abc.txt")

np.savetxt(rf"D:\Users\{USERNAME}\Downloads\test\a.txt", a, fmt='%d', delimiter=', ') # 改为保存为整数格式，逗号分隔；若是fmt=“%.8f”,则代表: 浮点数，8位小数
np.loadtxt(rf"D:\Users\{USERNAME}\Downloads\test\a.txt", delimiter=',')

四.文件对象file
a,b,c
Out[73]:
(array([[4, 3, 2],
        [3, 1, 3]]),
 array([1. , 0.2, 0.3]),
 array([0.84147098, 0.19866933, 0.29552021]))

with open(rf"D:\Users\{USERNAME}\Downloads\test\abc.npy", "wb") as f:
       np.save(f, a) # 按顺序保存到对象f
       np.save(f, b)
       np.save(f, c)
with open(rf"D:\Users\{USERNAME}\Downloads\test\abc.npy", "rb") as f:
       print(np.load(f))# 按顺序读取
       print(np.load(f))# 按顺序读取
       print(np.load(f))  # 按顺序读取

但若存储的是字典项时，加载时需注意： 
加载的数组实际上是一个scalar，这意味着它是一个具有表示“非数组”值的空形状的数组对象。特别是，是一个数据类型为object的标量，其中包含一个Python dict，而后者又在键'trajectories'下包含一个数值型NumPy数组。
在许多情况下，NumPy标量可以与它们包含的值模糊地使用(例如，标量可以像常规Python数一样使用)。然而，对于对象，情况要复杂得多，因为对象的方法不是通过NumPy标量公开的。要“解包”标量，可以使用item方法，该方法将获得“空的”内部值。然后，您将能够照常使用该对象。
>>>import numpy as np
>>>dt = np.load('trajectories.npy')
>>>dt
array({'trajectories': array([[[729.78449821, 391.1702509],
[912.41666667, 315.5       ],
[832.0577381 , 325.83452381]],
...,
[[852.92      , 174.16253968],
[923.36053131, 347.92694497],
[878.89942529, 323.26652299]]]), video_path: 'myPath', frames_per_second: 28}, dtype = object)
# 从numpy ndarray中提取字典项
dt.item()['trajectories']

####################################################################################
问题：numpy.uint16对象 如何迭代写入二进制文件
在Python中，你可以使用numpy.nditer对象来迭代numpy.uint16数组的元素，并使用struct模块将这些值写入二进制文件。以下是一个示例代码：

import numpy as np
import struct

# 创建一个包含uint16类型元素的numpy数组
arr = np.array([1, 2, 3, 4, 5], dtype=np.uint16)

# 打开一个二进制文件进行写入
with open('output.bin', 'wb') as f:
    # 使用nditer来迭代数组
    for value in np.nditer(arr):
        # 将uint16的值转换为2个字节，并写入文件
        f.write(struct.pack('H', value))
这段代码创建了一个包含uint16类型元素的数组，然后使用struct.pack函数将数组中的每个元素打包为2个字节，最后将这些字节写入名为output.bin的二进制文件中。struct.pack函数中的格式字符串'H'表示要写入的是无符号短整型（uint16），而'wb'模式确保了文件会以二进制形式被写入。

问题：numpy对象二进制文件，如何不一次全部加载到内存读取：
with open(file_name, 'r') as f:
    nbytes = f.seek(0, 2)
    flen = f.tell() // np.dtype('uint16').itemsize
data = np.memmap(file_name, dtype=np.dtype('uint16'), shape=(flen // max_length, max_length))


