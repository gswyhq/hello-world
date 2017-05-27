#!/usr/bin/python
# -*- coding:utf8 -*- #
from __future__ import  generators
from __future__ import  division
from __future__ import  print_function
from __future__ import  unicode_literals
import sys,os,json

if sys.version >= '3':
    PY3 = True
else:
    PY3 = False

if PY3:
    import pickle
else:
    import cPickle as pickle
    from codecs import open
    
import tensorflow as tf
# 创建一个常量op， 产生一个1x2矩阵，这个op被作为一个节点
# 加到默认视图中
# 构造器的返回值代表该常量op的返回值
matrix1 = tr.constant([[3., 3.]])

# 创建另一个常量op, 产生一个2x1的矩阵
matrix2 = tr.constant([[2.], [2.]])

# 创建一个矩阵乘法matmul op，把matrix1和matrix2作为输入：
product = tf.matmul(matrix1, matrix2)

#默认图现在有三个节点，两个constant() op和matmul() op。为了真正进行矩阵乘法的结果，你必须在会话里启动这个图。

# 启动默认图
sess = tf.Session()

# 调用sess的'run()' 方法来执行矩阵乘法op，传入'product'作为该方法的参数
# 上面提到，'product'代表了矩阵乘法op的输出，传入它是向方法表明，我们希望取回
# 矩阵乘法op的输出。
#
#整个执行过程是自动化的，会话负责传递op所需的全部输入。op通常是并发执行的。
#
# 函数调用'run(product)' 触发了图中三个op（两个常量op和一个矩阵乘法op）的执行。
# 返回值'result'是一个numpy 'ndarray'对象。
result = sess.run(product)
print result
# ==>[[12.]]

# 完成任务，关闭会话
sess.close()

#Session对象在使用完成后需要关闭以释放资源，除了显式调用close外，也可以使用with代码来自动完成关闭动作：
with tf.Session() as sess:
  result = sess.run([product])
  print result
  
#在实现上，Tensorflow将图形定义转换成分布式执行的操作，以充分利用可以利用的计算资源（如CPU或GPU）。一般你不需要显式指定使用CPU还是GPU，Tensorflow能自动检测。如果检测到GPU，Tensorflow会尽可能地使用找到的第一个GPU来执行操作。

#如果机器上有超过一个可用的GPU，除了第一个外的其他GPU是不参与计算的。为了让Tensorflow使用这些GPU，你必须将op明确地指派给它们执行。with...Device语句用来指派特定的CPU或GPU操作：

with tf.Session() as sess:
  with tf.device("/gpu:1"):
    matrix1 = tf.constant([[3., 3.]])
    matrix2 = tf.constant([[2.], [2.]])
    product = tf.matmul(matrix1, matrix2)

#设备用字符串进行标识，目前支持的设备包括：

#/cpu:0:机器的CPU
#/gpu:0:机器的第一个GPU，如果有的话
#/gpu:1:机器的的第二个GPU，以此类推

#为了便于使用诸如IPython之类的python交互环境，可以使用InteractiveSession代替Session类，使用Tensor.eval()和Operation.run()方法代替Session.run()。这样可以避免使用一个变量来持有会话：

# 进入一个交互式Tensorflow会话
import tensorflow as tf
sess = tf.InteractiveSession()

x = tf.Variable([1.0, 2.0])
a = tf.constant([3.0, 3.0]);

# 使用初始化器initializer op的run()方法初始化x
x.initializer.run()

# 增加一个减法sub op，从x减去a。运行减法op，输出结果
sud = tf.sub(x, a)
print sub.eval()
# ==>[-2. -1.]

#下面的例子演示了如何使用变量实现一个简单的计数器：

# 创建一个变量，初始为标量0
state = tf.Variable(0, name="counter")

# 创建一个op，其作用是使`state`增加1
one = tf.constant(1)
new_value = tf.add(state, one)
update = tf.assign(state, new_value)

# 启动图后，变量必须先经过init op初始化
# 首先先增加一个初始化op到图中
init_op = tf.initialize_all_variables()

# 启动图
with tf.Session() as sess:
  # 运行init op
  sess.run(init_op)
  # 打印 state 的初始值
  print sess.run(state)
  # 运行op， 更新state 并打印
  for _ in range(3):
    sess.run(update)
    print sess.run(state)

# 输出：
# 0
# 1
# 2
# 3
#代码中assign()操作是图所描述的表达式的一部分，正如add()操作一样，所以在调用run()执行表达式之前，它并不会真正执行赋值操作。

#为了取回操作的输出内容，可以在使用Session对象的run()调用执行图时，传入一些tensor，这些tensor会帮助你取回结果。在之前的例子里，我们只取回了单个节点state，但是你也可以取回多个tensor:

input1 = tf.constant(3.0)
input2 = tf.constant(4.0)
input3 = tf.constant(5.0)
intermed = tf.add(input2, input3)
mul = tf.mul(input1, intermed)

with tf.Session() as sess:
  result = sess.run([mul, intermed])
  print result

# print
# [27.0, 9.0]

#需要获得更多个tensor值，在op的依次运行中获得（而不是逐个去获得tenter）

上述示例在计算图中引入tensor，以常量或变量的形式存储。Tensorflow还提供了feed机制，该机制可以临时替代图中的任意操作中的tensor可以对图中任何操作提交补丁，直接插入一个tensor。

feed使用一个tensor值临时替换一个操作的输出结果，你可以提供feed数据作为run()调用的参数。
feed只在调用它的方法内有效，方法结束，feed就会消失。最常见的用例是将某些特殊的操作指定为feed操作，标记的方法是使用tf.placeholder()为这些操作创建占位符。

input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.mul(input1, input2)

with tf.Session() as sess:
  print sess.run([output], feed_dict={input:[7.]， input2:[2.]})

# print
# [array([ 14.], dtype=float32)]

def main():
    pass


if __name__ == "__main__":
    main()
    
#http://blog.csdn.net/u014595019/article/details/54093161
#http://blog.csdn.net/jdbc/article/details/51874010

