#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
from tensorflow.python.keras.layers.preprocessing.text_vectorization import LOWER_AND_STRIP_PUNCTUATION, SPLIT_ON_WHITESPACE, INT
from keras.engine.base_preprocessing_layer import version_utils

vocab_data = ['', '[UNK]', '是', '我', '国', '人', '中', '个', '一', '？', '话', '好', '听', '吗', '你', "[SEP]",]
text_layer = tf.keras.layers.TextVectorization(
    max_tokens=5000,  # 词汇表最大尺寸
    output_mode='int',  # 输出整数索引
    vocabulary=vocab_data,  # 可选词典
)  # 创建 TextVectorization 层
print(text_layer)
# <keras.layers.preprocessing.text_vectorization.TextVectorization object at 0x0000015FADCE5E48>
data = [
    "听 话",  # 第1句话
    "你 好 吗 ？",  # 第2句话
    "我 是 一 个 中 国 人" , # 第3句话
    "一 个 中 国 人 是 我"
]  # 数据
text_layer.adapt(data)  # 数据加入 TextVectorization 层
text_layer.get_vocabulary()  # 得到所有单词字典（字典里多了 '' '[UNK]'）
print({index: w for index, w in enumerate(text_layer.get_vocabulary() )})
# {0: '', 1: '[UNK]', 2: '是', 3: '我', 4: '国', 5: '人', 6: '中', 7: '个', 8: '一', 9: '？', 10: '话', 11: '好', 12: '听', 13: '吗', 14: '你'}

text_layer(data)  # 得到 data 中字典下标组成的数组
# Out[67]:
# <tf.Tensor: shape=(4, 7), dtype=int64, numpy=
# array([[12, 10,  0,  0,  0,  0,  0],
#        [14, 11, 13,  9,  0,  0,  0],
#        [ 3,  2,  8,  7,  6,  4,  5],
#        [ 8,  7,  6,  4,  5,  2,  3]], dtype=int64)>


tf.keras.layers.TextVectorization(
    max_tokens=None,  # 词汇表的最大值，若设置为None则表示不限制其大小；若词汇数大于了最大值，则会将出现次数最少的词丢掉；注意，该最大值是包含了oov令牌数量的，所以最大词汇量是max_tokens-size(oov)的大小。
    standardize=LOWER_AND_STRIP_PUNCTUATION,  # 标准化方式，默认是LOWER_AND_STRIP_PUNCTUATION，即转小写并去掉标点符号；可以自定义Callable函数。
    split=SPLIT_ON_WHITESPACE, # 分词器，默认是按空格分词；可以自定义分词方式。
    ngrams=None,  # 是否创建n-grams，参数可以是None或者任意整数，None代表不创建。
    output_mode=INT,  # 参数可以是"int"，“binary”，“count"和"tf-idf”；当设置为int时，输出为token索引列表；当设置为binary时，输出为one-hot编码；当设置为count时，类似binary，不同在于数组的每一位上表示词的统计个数；当设置为tf-idf时，类似binary，不同在于数组每一位上的个数表示该词在其他文本中出现的频次。
    output_sequence_length=None, # 只有output_mode设置为int时才有效；参数表示输出向量的维度，其维度会根据设置的长度进行填充或删减
    pad_to_max_tokens=True, # 只有output_mode设置为binary、count和tf-idf时才有效；如果设置为True，即使词汇表的token数少于max_tokens，输入的特征向量也会被填充至max_tokens的大小
    vocabulary=None, # 设置词汇表；参数可以是词汇list也可以是词汇表所在的文件路径；注意词汇表中的token不能重复，否则会报错
)

# 测试数据
text_dataset = tf.data.Dataset.from_tensor_slices(list('''中国第三艘航空母舰下水，命名福建舰，航空母舰配置电磁弹射和阻拦装置，满载排水量8万余吨。'''))
max_features = 5000  # 最大词汇表大小
max_len = 4  # 词向量维度
embedding_dims = 2

# 创建TextVectorization
vectorize_layer = TextVectorization(
 max_tokens=max_features,
 output_mode='int',
 output_sequence_length=max_len)

# 创建词汇表，对于大数据集，使用批量输入是最好的方式；
# 若已有词汇表，则可以调用vectorize_layer.set_vocabulary(vocab)直接设置
vectorize_layer.adapt(text_dataset.batch(64))

# 创建模型
model = tf.keras.models.Sequential()

# 小批量输入，每批次包含32条string类型的数组
model.add(tf.keras.Input(shape=(1,), dtype=tf.string))

# 将TextVectorization作为第一层
# 通过这一层后，会输出形如(batch_size, max_len)的张量
model.add(vectorize_layer)

# 现在模型具有了将字符串转为词向量的能力
# 随后便可以将词向量输入进词嵌入层进行下一层的学习了
input_data = [["你 好 福 建 舰"], ["航 空 母 舰 就 是 烧 钱 的 玩 意"]]
model.predict(input_data)
# array([[ 1,  1, 16, 23],
#        [ 4,  6,  8,  3]], dtype=int64)


def main():
    pass


if __name__ == '__main__':
    main()
