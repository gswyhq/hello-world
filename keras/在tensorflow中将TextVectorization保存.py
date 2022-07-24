#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pickle
import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from keras.models import load_model
from keras.preprocessing import sequence

text_dataset = tf.data.Dataset.from_tensor_slices(['什 么 是 正 则 化 ？', '简 单 来 说 ， 正 则 化 就 是 一 个 减 少 过 拟 合 的 过 程 。',
                                                   '它 是 当 模 型 包 含 噪 音 时 ， 向 模 型 学 习 过 程 引 入 警 告 的 一 种 数 学 方 法 。',
                                                   '实 际 上 ， 它 是 一 种 在 过 拟 合 情 况 下 对 模 型 权 重 进 行 惩 罚 的 方 法 。',
                                                   '在 深 度 学 习 中 ， 神 经 元 连 接 的 权 重 会 在 每 次 迭 代 后 更 新 。',
                                                   '当 模 型 遇 到 一 个 有 噪 声 样 本 、 并 假 设 该 样 本 是 有 效 样 本 时 ， 它 会 尝 试 通 过 更 新 与 噪 声 一 致 的 权 重 来 适 应 模 型 。',
                                                   '在 真 实 的 样 本 数 据 中 ， 噪 声 数 据 点 与 常 规 数 据 大 相 径 庭 。', '因 此 ， 权 重 的 更 新 也 将 与 噪 声 同 步 （ 权 重 的 变 化 是 巨 大 的 ） 。',
                                                   '正 则 化 过 程 将 边 的 权 重 加 入 已 定 义 的 损 失 函 数 中 ， 整 体 上 表 示 更 高 的 损 失 。',
                                                   '然 后 ， 神 经 网 络 通 过 自 我 调 优 以 降 低 损 失 ， 使 权 重 朝 着 正 确 的 方 向 更 新 ； 这 是 通 过 在 学 习 过 程 中 忽 略 噪 声 、 而 不 是 适 应 噪 声 来 实 现 的 。',
                                                   '正 则 化 的 过 程 可 以 表 示 为 ： 损 失 函 数 = 损 失 （ 针 对 模 型 定 义 的 ） + 超 参 数 × [ 权 重 ] 。', '其 中 ， 超 参 数 由 λ / 2 m 表 示 ， 其 中 λ 的 值 由 用 户 定 义 。',
                                                   '基 于 权 重 加 入 损 失 函 数 的 方 式 ， 存 在 两 种 不 同 类 型 的 正 则 化 技 术 ： L 1 正 则 化 和 L 2 正 则 化 。']
)
# 训练 TextVectorization 层
vectorizer = TextVectorization(max_tokens=32, output_mode='tf-idf',ngrams=None)
vectorizer.adapt(text_dataset.batch(1024))

# 获取某个词的向量
print (vectorizer("真"))

tv_layer_save_file = r"D:\Users\{}\Downloads\test\tv_layer.pkl".format(os.getenv("USERNAME"))

# 保存TextVectorization层的参数及向量
pickle.dump({'config': vectorizer.get_config(),
             'weights': vectorizer.get_weights()}
            , open(tv_layer_save_file, "wb"))

print ("*"*10)
# 接下来需要使用保存的参数创建一个对象，并加载保存的参数

from_disk = pickle.load(open(tv_layer_save_file, "rb"))
new_v = TextVectorization.from_config(from_disk['config'])

# 你必须用一些虚拟数据调用 `adapt`（Keras 中的 BUG）
new_v.adapt(tf.data.Dataset.from_tensor_slices(["xyz"]))
new_v.set_weights(from_disk['weights'])

# 再次查看对应单词的向量
print (new_v("真"))


# 当然也可以通过下面这样，构建个模型，再将模型保存为hdf5 或转换为PB文件；
text_dataset = tf.data.Dataset.from_tensor_slices(["foo", "bar", "baz"])
max_features = 5000  # Maximum vocab size.
max_len = 4  # Sequence length to pad the outputs to.
vectorize_layer = tf.keras.layers.TextVectorization(
 max_tokens=max_features,
 output_mode='int',
 output_sequence_length=max_len)
vectorize_layer.adapt(text_dataset.batch(64))
model = tf.keras.models.Sequential()
model.add(tf.keras.Input(shape=(1,), dtype=tf.string))
model.add(vectorize_layer)
# model.save("model.hdf5")  # 此处会报错
model.save("./result/vectorize_layer_max_length_128_max_tokens_1000.model", save_format='tf')

input_data = [["foo qux bar"], ["qux baz"]]
model.predict(input_data)

model2 = load_model("./result/vectorize_layer_max_length_128_max_tokens_1000.model")
model2.predict(input_data)

# 但输出的维度不对，可能需要在后面补零；
# sequence.pad_sequences([list(text_vectorization(text).numpy()) for text in t[0]], padding="post", maxlen=128, value=0)

def main():
    pass


if __name__ == '__main__':
    main()

