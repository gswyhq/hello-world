#!/usr/bin/python3
# coding: utf-8

# 目前TF的RNN APIs主要集中在tensorflow.models.rnn中的rnn和rnn_cell两个模块。
# 其中，后者定义了一些常用的RNN cells，包括RNN和优化的LSTM、GRU等等；前者则提供了一些helper方法。
# 创建一个基础的RNN很简单：

from tensorflow.models.rnn import rnn_cell

cell = rnn_cell.BasicRNNCell(inputs, state)
# 创建一个LSTM或者GRU的cell？

cell = rnn_cell.BasicLSTMCell(num_units)  #最最基础的，不带peephole。
cell = rnn_cell.LSTMCell(num_units, input_size)  #可以设置peephole等属性。
cell = rnn_cell.GRUCell(num_units)

# 调用呢？
output, state = cell(input, state)
# 这样自己按timestep调用需要设置variable_scope的reuse属性为True，懒人怎么做，TF也给想好了：


state = cell.zero_state(batch_size, dtype=tf.float32)
outputs, states = rnn.rnn(cell, inputs, initial_state=state)

# 再懒一点：
outputs, states = rnn.rnn(cell, inputs, dtype=tf.float32)
# 怕overfit，加个Dropout如何？


cell = rnn_cell.DropoutWrapper(cell, input_keep_prob=0.5, output_keep_prob=0.5)

# 做个三层的带Dropout的网络？
cell = rnn_cell.DropoutWrapper(cell, output_keep_prob=0.5)
cell = rnn_cell.MultiRNNCell([cell] * 3)

inputs = tf.nn.dropout(inputs, 0.5)  #给第一层单独加个Dropout。
# 一个坑——用rnn.rnn要按照timestep来转换一下输入数据，比如像这样：


inputs = [tf.reshape(t, (input_dim[0], 1)) for t in tf.split(1, input_dim[1], inputs)]
# rnn.rnn()的输出也是对应每一个timestep的，如果只关心最后一步的输出，取outputs[-1]即可。

# 注意一下子返回值的dimension和对应关系，损失函数和其它情况没有大的区别。


def main():
    pass


if __name__ == '__main__':
    main()