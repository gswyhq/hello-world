#!/usr/bin/python3
# coding: utf-8

import tensorflow as tf

# TensorFlow内部的默认图是存在_default_graph_stack里了，但是我们不能直接使用。我们使用tf.get_default_graph()来使用它。
graph = tf.get_default_graph()

# TensorFlow图里的节点被称为“运算”（operation）或“ops”。我们可以通过graph.get_operations()来得到图里的所有的运算。
graph.get_operations()
## []

# 现在图里什么都没有。我们需要往图里放入一些我们希望TensorFlow的图里计算的东西。让我们从一个简单的常量输入开始。
input_value = tf.constant(1.0)

# 现在这个常量已经作为图里的一个节点（一个运算）存在了。这个Python变量input_value间接地指向了这个运算，但是我们也能在这个默认图里面找到。
operations = graph.get_operations()
# In[25]: operations
# Out[25]: [<tensorflow.python.framework.ops.Operation at 0x7f1b5df7d4a8>]
# In[26]: operations[0]
# Out[26]: <tensorflow.python.framework.ops.Operation at 0x7f1b5df7d4a8>
# In[27]: operations[0].name
# Out[27]: 'Const'
# In[28]: operations[0].node_def
# Out[28]:
# name: "Const"
# op: "Const"
# attr {
#   key: "dtype"
#   value {
#     type: DT_FLOAT
#   }
# }
# attr {
#   key: "value"
#   value {
#     tensor {
#       dtype: DT_FLOAT
#       tensor_shape {
#       }
#       float_val: 1.0
#     }
#   }
# }

# 仔细看看我们的input_value，就会发现这个是一个无维度的32位浮点张量：就是一个数字。
# In[21]: input_value
# Out[21]: <tf.Tensor 'Const:0' shape=() dtype=float32>
# 值得注意的是，这个结果并没有说明这个数字是多少？为了执行input_value这句话，并给出这个数字的值，我们需要创造一个“会话”。让图里的计算在其中执行并明确地要执行input_value并给出结果（会话会默认地去找那个默认图）
sess = tf.Session()
sess.run(input_value)
## 1.0

weight = tf.Variable(0.8)

# 查看所有的这些运算的名字。
for op in graph.get_operations():
    print(op.name)

output_value = weight * input_value

# 现在图里有六个运算，最后一个是相乘。
op = graph.get_operations()[-1]
# op.name
## 'mul'
for op_input in op.inputs:
    print(op_input)

# 怎么才能看到乘积是多少？我们必须“运行”这个output_value运算。但是这个运算依赖于一个变量：权重。我们告诉TensorFlow这个权重的初始值是0.8，但在这个会话里，这个值还没有被设置。tf.initialize_all_variables()函数生成了一个运算，来初始化所有的变量（我们的情况是只有一个变量）。随后我们就可以运行这个运算了。
init = tf.global_variables_initializer()
sess.run(init)
# tf.global_variables_initializer()的结果会包括现在图里所有变量的初始化器。所以如果你后续加入了新的变量，你就需要再次使用tf.global_variables_initializer()。一个旧的init是不会包括新的变量的。

# 现在我们已经准备好运行output_value运算了。
sess.run(output_value)
## 0.80000001

# 在TensorBoard里查看你的图
# 到目前为止，我们的图是很简单的，但是能看到她的图形表现形式也是很好的。我们用TensorBoard来生成这个图形。TensorBoard读取存在每个运算里面的名字字段，这和Python里的变量名是很不一样的。我们可以使用这些TensorFlow的名字，并转成更方便的Python变量名。这里tf.mul和我前面使用*来做乘运算是等价的，但这个操作可以让我们设置运算的名字。
x = tf.constant(1.0, name='input')
w = tf.Variable(0.8, name='weight')
y = tf.mul(w, x, name='output')
# TensorBoard是通过查看一个TensorFlow会话创建的输出的目录来工作的。我们可以先用一个SummaryWriter来写这个输出。如果我们只是创建一个图的输出，它就将图写出来。

# 构建SummaryWritery已被弃用，改用（FileWriter）的第一个参数是一个输出目录的名字。如果此目录不存在，则在构建SummaryWriter时会被建出来。即在当前运行目录下生成一个名为“log_simple_graph”的目录
summary_writer = tf.summary.FileWriter('log_simple_graph', sess.graph)
# 现在我们可以通过命令行来启动TensorBoard了。

# $ tensorboard --logdir log_simple_graph

# TensorBoard会运行一个本地的Web应用，端口6006(6006是goog这个次倒过的对应)。在你本机的浏览器里登陆lhttp://127.0.1.1:6006/#graphs，你就可以看到在TensorFlow里面创建的图

# 用实际输出和期望输出之差的平方来作为损失的测量值。
y_ = tf.constant(0.0)
loss = (y - y_)**2

# 使用梯度下降优化器来基于损失值的导数去更新权重。这个优化器采用一个学习比例来调整每一步更新的大小。这里我们设为0.025。
optim = tf.train.GradientDescentOptimizer(learning_rate=0.025)

grads_and_vars = optim.compute_gradients(loss)
sess.run(tf.global_variables_initializer())  # initialize_all_variables，被弃用，改用：global_variables_initializer
sess.run(grads_and_vars[1][0])

# 运用这个梯度来完成反向传播。
sess.run(optim.apply_gradients(grads_and_vars))
sess.run(w)
## 0.75999999  # about 0.76
# 现在权重减少了0.04，这是因为优化器减去了梯度乘以学习比例（1.6*0.025）。权重向着正确的方向在变化。

# 其实我们不必像这样调用优化器。我们可以形成一个运算，自动地计算和使用梯度：train_step。
train_step = tf.train.GradientDescentOptimizer(0.025).minimize(loss)
for i in range(100):
    sess.run(train_step)
    sess.run(y)

# 想知道每次训练步骤后，系统都是怎么去预测输出的。为此，我们可以在训练循环里面打印输出值。
sess.run(tf.global_variables_initializer())
for i in range(100):
    print('before step {}, y is {}'.format(i, sess.run(y)))
    sess.run(train_step)

## before step 0, y is 0.800000011921
## before step 1, y is 0.759999990463
## ...
## before step 98, y is 0.00524811353534
## before step 99, y is 0.00498570781201

# 通过加入能总结图自己状态的运算来提交给计算图。这里我们会创建一个运算，它能报告y的当前值，即神经元的输出。
summary_y = tf.summary.scalar('output', y)  # scalar_summary，被弃用
# 当你运行一个总结运算，它会返回给一个protocal buffer文本的字符串。用SummaryWriter可以把这个字符串写入一个日志目录。
summary_writer = tf.summary.FileWriter('log_simple_stats')
sess.run(tf.global_variables_initializer())
for i in range(100):
    summary_str = sess.run(summary_y)
    summary_writer.add_summary(summary_str, i)
    sess.run(train_step)

# 在运行命令  tensorboard --logdir log_simple_stats 后，你就可以在localhost:6006/#events里面看到一个可交互的图形

# 下面是代码的完全版。它相当的小。但每个小部分都显示了有用且可理解的TensorflowFlow的功能。


x = tf.constant(1.0, name='input')
w = tf.Variable(0.8, name='weight')
y = tf.mul(w, x, name='output')
y_ = tf.constant(0.0, name='correct_value')
loss = tf.pow(y - y_, 2, name='loss')
train_step = tf.train.GradientDescentOptimizer(0.025).minimize(loss)

for value in [x, w, y, y_, loss]:
    tf.summary.scalar(value.op.name, value)

summaries = tf.summary.merge_all()

sess = tf.Session()
summary_writer = tf.summary.FileWriter('log_simple_stats', sess.graph)

sess.run(tf.global_variables_initializer())
for i in range(100):
    summary_writer.add_summary(sess.run(summaries), i)
    sess.run(train_step)