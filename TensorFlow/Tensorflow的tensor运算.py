#!/usr/bin/python3
# coding: utf-8

import tensorflow as tf
import numpy as np

x = 2
y = 3
add_op = tf.add(x, y)  # 加法
mul_op = tf.multiply(x, y)  # 乘法
useless = tf.multiply(x, add_op) # x*(x+y)
pow_op = tf.pow(add_op, mul_op)  # 幂次方, (x+y)**(x*y)
with tf.Session() as sess:
    z, not_useless = sess.run([pow_op, useless])
    print(z, not_useless)

def relu(x):  # 要构造一个和x shape一样的Tensor。源码中应该不会用效率这么低的写法。
    y = tf.constant(0.0, shape=x.get_shape())
    return tf.where(tf.greater(x, y), x, y)


with tf.Session() as sess:
    x = tf.Variable(tf.random_normal(shape=[10], stddev=10))
    sess.run(tf.global_variables_initializer())
    x_relu = relu(x)
    data_x, data_x_relu = sess.run((x, x_relu))
    for i in range(0, len(data_x)):
        print("%.5f  --relu--> %.5f" % (data_x[i], data_x_relu[i]))
    
def main():
    pass


if __name__ == '__main__':
    main()