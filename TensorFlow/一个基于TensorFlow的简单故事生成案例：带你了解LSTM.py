#!/usr/bin/python3
# coding: utf-8

# http://www.jiqizhixin.com/article/2716

import numpy as np
import tensorflow as tf
# from tensorflow.contrib import rnn
# from tensorflow.models.rnn import rnn_cell
from tensorflow.python.ops import rnn, rnn_cell
import random
import collections
import time


# 取自伊索寓言的短故事，其中有 112 个不同的符号。单词和标点符号都视作符号。
content = """long ago , the mice had a general council to consider what measures they could take to outwit their common enemy , the cat . some said this , and some said that but at last a young mouse got up and said he had a proposal to make , which he thought would meet the case . you will all agree , said he , that our chief danger consists in the sly and treacherous manner in which the enemy approaches us . now , if we could receive some signal of her approach , we could easily escape from her . i venture , therefore , to propose that a small bell be procured , and attached by a ribbon round the neck of the cat . by this means we should always know when she was about , and could easily retire while she was in the neighbourhood . this proposal met with general applause , until an old mouse got up and said that is all very well , but who is to bell the cat ? the mice looked at one another and nobody spoke . then the old mouse said it is easy to propose impossible remedies .
"""
training_data = np.reshape(content.split(), [-1, ])

# 将文本中的 3 个符号以正确的序列输入 LSTM，以 1 个标记了的符号作为输出，最终神经网络将学会正确地预测下一个符号
# 即有 3 个输入和 1 个输出的 LSTM 单元

# 严格说来，LSTM 只能理解输入的实数。一种将符号转化为数字的方法是基于每个符号出现的频率为其分配一个对应的整数。
# 例如，上面的短文中有 112 个不同的符号。如列表 2 所示的函数建立了一个有如下条目
# [「,」: 0 ] [「the」: 1 ], …, [「council」: 37 ],…,[「spoke」= 111 ] 的词典。
# 而为了解码 LSTM 的输出，同时也生成了逆序字典。

# 类似地，预测值也是一个唯一的整数值与逆序字典中预测符号的索引相对应。例如：如果预测值是 37，预测符号便是「council」。

# 输出的生成看起来似乎简单，但实际上 LSTM 为下一个符号生成了一个含有 112 个元素的预测概率向量，
# 并用 softmax() 函数归一化。有着最高概率值的元素的索引便是逆序字典中预测符号的索引值（例如：一个 one-hot 向量）。

# 每一个输入符号被分配给它的独一无二的整数值所替代。输出是一个表明了预测符号在反向词典中索引的 one-hot 向量。

# Target log path
logs_path = '/tmp/tensorflow/rnn_words'
# writer = tf.summary.FileWriter(logs_path)

start_time = time.time()
def elapsed(sec):
    if sec<60:
        return str(sec) + " sec"
    elif sec<(60*60):
        return str(sec/60) + " min"
    else:
        return str(sec/(60*60)) + " hr"


# 建立字典和逆序字典的函数
def build_dataset(words):
    count = collections.Counter(words).most_common()
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return dictionary, reverse_dictionary


dictionary, reverse_dictionary = build_dataset(training_data)
vocab_size = len(dictionary)
# 网络的常量、权值和偏差设置如下：
learning_rate = 0.001
training_iters = 50000
display_step = 1000
n_input = 3

# LSTM单元数
n_hidden = 512

# tf Graph input
x = tf.placeholder("float", [None, n_input, 1])
y = tf.placeholder("float", [None, vocab_size])

# RNN 输出节点的权重和偏差
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, vocab_size]))
}
biases = {
    'out': tf.Variable(tf.random_normal([vocab_size]))
}

def RNN(x, weights, biases):

    # reshape to [1, n_input]
    x = tf.reshape(x, [-1, n_input])

    # 每一个输入符号被分配给它的独一无二的整数值所替代
    # (eg. [had] [a] [general] -> [20] [6] [33])
    x = tf.split(1,n_input,x)

    # LSTM 的精度可以通过增加层来改善。
    # 两层
    cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(n_hidden), tf.nn.rnn_cell.BasicLSTMCell(n_hidden)])

    # 1-layer LSTM with n_hidden units.
    # rnn_cell = rnn.BasicLSTMCell(n_hidden)

    # 预测
    outputs, states = rnn.rnn(cell, x, dtype=tf.float32)

    # 有 n_input 个输出结果，但仅仅取最后一个
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

pred = RNN(x, weights, biases)

# Loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)

# Model evaluation
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()



# Launch the graph
with tf.Session() as session:
    session.run(init)
    step = 0
    offset = random.randint(0,n_input+1)
    end_offset = n_input + 1
    acc_total = 0
    loss_total = 0

    # writer.add_graph(session.graph)

    # 训练过程中的优化
    # 精度和损失被累积以监测训练过程。通常 50,000 次迭代足以达到可接受的精度要求。
    while step < training_iters:
        # Generate a minibatch. Add some randomness on selection process.
        if offset > (len(training_data)-end_offset):
            offset = random.randint(0, n_input+1)

        # 训练过程中的每一步，3 个符号都在训练数据中被检索。然后 3 个符号转化为整数以形成输入向量。
        symbols_in_keys = [ [dictionary[ str(training_data[i])]] for i in range(offset, offset+n_input) ]
        symbols_in_keys = np.reshape(np.array(symbols_in_keys), [-1, n_input, 1])

        # 将符号转化为整数向量作为输入
        # 训练标签是一个位于 3 个输入符号之后的 one-hot 向量。
        symbols_out_onehot = np.zeros([vocab_size], dtype=float)
        symbols_out_onehot[dictionary[str(training_data[offset+n_input])]] = 1.0
        symbols_out_onehot = np.reshape(symbols_out_onehot,[1,-1])

        # 单向量作为标签
        # 在转化为输入词典的格式后，进行如下的优化过程：
        # 一个训练间隔的预测和精度数据示例（间隔 1000 步）
        # 代价是标签和 softmax() 预测之间的交叉熵，它被 RMSProp 以 0.001 的学习率进行优化。在本文示例的情况中，RMSProp 通常比 Adam 和 SGD 表现得更好。
        _, acc, loss, onehot_pred = session.run([optimizer, accuracy, cost, pred], \
                                                feed_dict={x: symbols_in_keys, y: symbols_out_onehot})
        loss_total += loss
        acc_total += acc
        if (step+1) % display_step == 0:
            print("Iter= " + str(step+1) + ", Average Loss= " + \
                  "{:.6f}".format(loss_total/display_step) + ", Average Accuracy= " + \
                  "{:.2f}%".format(100*acc_total/display_step))
            acc_total = 0
            loss_total = 0
            symbols_in = [training_data[i] for i in range(offset, offset + n_input)]
            symbols_out = training_data[offset + n_input]
            symbols_out_pred = reverse_dictionary[int(tf.argmax(onehot_pred, 1).eval())]
            print("%s - [%s] vs [%s]" % (symbols_in,symbols_out,symbols_out_pred))
        step += 1
        offset += (n_input+1)
    print("Optimization Finished!")
    print("Elapsed time: ", elapsed(time.time() - start_time))
    print("Run on command line.")
    print("\ttensorboard --logdir=%s" % (logs_path))
    print("Point your web browser to: http://localhost:6006/")
    while True:
        prompt = "%s words: " % n_input
        sentence = input(prompt)
        sentence = sentence.strip()
        words = sentence.split(' ')
        if len(words) != n_input:
            continue
        try:
            symbols_in_keys = [dictionary[str(words[i])] for i in range(len(words))]
            for i in range(32):
                keys = np.reshape(np.array(symbols_in_keys), [-1, n_input, 1])
                onehot_pred = session.run(pred, feed_dict={x: keys})
                onehot_pred_index = int(tf.argmax(onehot_pred, 1).eval())
                sentence = "%s %s" % (sentence,reverse_dictionary[onehot_pred_index])
                symbols_in_keys = symbols_in_keys[1:]
                symbols_in_keys.append(onehot_pred_index)
            print(sentence)
        except:
            print("Word not in dictionary")




# 损失和优化器


# rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(n_hidden),rnn.BasicLSTMCell(n_hidden)])


def main():
    pass


if __name__ == '__main__':
    main()