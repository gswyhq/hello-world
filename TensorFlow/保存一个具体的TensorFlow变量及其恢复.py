#!/usr/bin/python3
# coding: utf-8

import tensorflow as tf

def save_model():

    w1 = tf.Variable(tf.truncated_normal(shape=[10]), name='w1')
    w2 = tf.Variable(tf.truncated_normal(shape=[20]), name='w2')
    tf.add_to_collection('vars', w1)
    tf.add_to_collection('vars', w2)
    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver.save(sess, 'my-model')
    # `save` 方法会隐式调用 `export_meta_graph`.
    # 会生成图形文件:my-model.meta

def restore_model():

    sess = tf.Session()
    new_saver = tf.train.import_meta_graph('my-model.meta')
    new_saver.restore(sess, tf.train.latest_checkpoint('./'))
    all_vars = tf.get_collection('vars')
    for v in all_vars:
        v_ = sess.run(v)
        print(v_)

def main():
    # save_model()  # 保存变量
    restore_model()  # 恢复变量


if __name__ == '__main__':
    main()
