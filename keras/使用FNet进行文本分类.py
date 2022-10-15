#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# https://keras.io/examples/nlp/fnet_classification_with_keras_nlp/
# 自注意力机制是一项昂贵的操作，时间复杂度为O(n^2)，其中n是输入中的令牌数。因此，人们一直在努力降低自注意力机制的时间复杂度并在不牺牲结果质量的情况下提高性能。
#
# 2020 年，一篇题为 FNet: Mixing Tokens with Fourier Transforms的论文 将 BERT 中的 self-attention 层替换为一个简单的 Fourier Transform 层，用于“token 混合”。这导致了相当的准确性和训练期间的加速。特别是，论文中的几点很突出：
#
# 作者声称 FNet 在 GPU 上比 BERT 快 80%，在 TPU 上快 70%。这种加速的原因有两个：a）傅里叶变换层是未参数化的，它没有任何参数，b）作者使用快速傅里叶变换（FFT）；这将时间复杂度从O(n^2) （在自注意力的情况下）降低到O(n log n).
# FNet 设法在 GLUE 基准上实现了 BERT 92-97% 的准确度。

import keras_nlp
import random
import tensorflow as tf
import os

from tensorflow import keras
from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert_vocab

BATCH_SIZE = 64
EPOCHS = 3
MAX_SEQUENCE_LENGTH = 512
VOCAB_SIZE = 15000

EMBED_DIM = 128
INTERMEDIATE_DIM = 512

# !wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
# !tar -xzf aclImdb_v1.tar.gz

# 样本以文本文件的形式存在。让我们检查目录的结构。

print(os.listdir(r"D:\Users\{}\data\aclImdb".format(os.getenv('USERNAME'))))
print(os.listdir(r"D:\Users\{}\data\aclImdb\train".format(os.getenv('USERNAME'))))
print(os.listdir(r"D:\Users\{}\data\aclImdb\test".format(os.getenv('USERNAME'))))



# 该目录包含两个子目录：train和test. 每个子目录又包含两个文件夹：pos分别neg用于正面和负面评论。在我们加载数据集之前，让我们删除该./aclImdb/train/unsup 文件夹，因为它有未标记的样本。

# !rm -rf aclImdb/train/unsup
# 我们将使用该实用程序从文本文件keras.utils.text_dataset_from_directory生成我们的标记数据集。tf.data.Dataset

train_ds = keras.utils.text_dataset_from_directory(
    r"D:\Users\{}\data\aclImdb\train".format(os.getenv('USERNAME')),
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset="training",
    seed=42,
)
val_ds = keras.utils.text_dataset_from_directory(
    r"D:\Users\{}\data\aclImdb\train".format(os.getenv('USERNAME')),
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset="validation",
    seed=42,
)
test_ds = keras.utils.text_dataset_from_directory(r"D:\Users\{}\data\aclImdb\train".format(os.getenv('USERNAME')), batch_size=BATCH_SIZE)

# train_ds.class_names
# Out[26]: ['neg', 'pos']

# 我们现在将文本转换为小写。
train_ds = train_ds.map(lambda x, y: (tf.strings.lower(x), y))
val_ds = val_ds.map(lambda x, y: (tf.strings.lower(x), y))
test_ds = test_ds.map(lambda x, y: (tf.strings.lower(x), y))

# 让我们打印一些样本。
for text_batch, label_batch in train_ds.take(1):
    for i in range(3):
        print(text_batch.numpy()[i])
        print(label_batch.numpy()[i])

# 标记数据
# 我们将使用该keras_nlp.tokenizers.WordPieceTokenizer层来标记文本。keras_nlp.tokenizers.WordPieceTokenizer采用 WordPiece 词汇表，并具有对文本进行标记和对标记序列进行去标记的功能。
#
# 在我们定义分词器之前，我们首先需要在我们拥有的数据集上对其进行训练。WordPiece 分词算法是一种子词分词算法；在语料库上训练它为我们提供了一个子词词汇表。子词标记器是单词标记器（单词标记器需要非常大的词汇表才能很好地覆盖输入单词）和字符标记器（字符并不像单词那样真正编码含义）之间的折衷。幸运的是，TensorFlow Text 使得在语料库上训练 WordPiece 变得非常简单，如 本指南中所述。
#
# 注意：FNet 的官方实现使用 SentencePiece Tokenizer。

def train_word_piece(ds, vocab_size, reserved_tokens):
    bert_vocab_args = dict(
        # The target vocabulary size
        vocab_size=vocab_size,
        # Reserved tokens that must be included in the vocabulary
        reserved_tokens=reserved_tokens,
        # Arguments for `text.BertTokenizer`
        bert_tokenizer_params={"lower_case": True},
    )

    # Extract text samples (remove the labels).
    word_piece_ds = ds.unbatch().map(lambda x, y: x)
    vocab = bert_vocab.bert_vocab_from_dataset(
        word_piece_ds.batch(1000).prefetch(2), **bert_vocab_args
    )
    return vocab

# 每个词汇表都有一些特殊的保留标记。我们有两个这样的令牌：
#
# "[PAD]"- 填充令牌。当输入序列长度小于最大序列长度时，将填充标记附加到输入序列长度。
# "[UNK]"- 未知令牌。
reserved_tokens = ["[PAD]", "[UNK]"]
train_sentences = [element[0] for element in train_ds]
vocab = train_word_piece(train_ds, VOCAB_SIZE, reserved_tokens)

# 让我们看看一些令牌！

print("Tokens: ", vocab[100:110])
# Tokens:  ['in', 'this', 'that', 'was', 'as', 'for', 'movie', 'with', 'but', 'film']
# 现在，让我们定义分词器。我们将使用上面训练的词汇表配置分词器。我们将定义一个最大序列长度，以便如果序列的长度小于指定的序列长度，则所有序列都被填充到相同的长度。否则，序列被截断。

# tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(
#     vocabulary=vocab,
#     lowercase=False,
#     sequence_length=MAX_SEQUENCE_LENGTH,
# )

tokenizer = keras_nlp.tokenizers.Tokenizer(
    vocabulary=vocab,
    lowercase=False,
    sequence_length=MAX_SEQUENCE_LENGTH,
)
# 让我们尝试对数据集中的样本进行标记！为了验证文本是否被正确标记化，我们还可以将标记列表解标记回原始文本。

input_sentence_ex = train_ds.take(1).get_single_element()[0][0]
input_tokens_ex = tokenizer(input_sentence_ex)

print("Sentence: ", input_sentence_ex)
print("Tokens: ", input_tokens_ex)
print("Recovered text after detokenizing: ", tokenizer.detokenize(input_tokens_ex))

# 格式化数据集
# 接下来，我们将以将馈送到模型的形式格式化我们的数据集。我们需要对文本进行标记。

def format_dataset(sentence, label):
    sentence = tokenizer(sentence)
    return ({"input_ids": sentence}, label)


def make_dataset(dataset):
    dataset = dataset.map(format_dataset, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.shuffle(512).prefetch(16).cache()


train_ds = make_dataset(train_ds)
val_ds = make_dataset(val_ds)
test_ds = make_dataset(test_ds)


# 构建模型
# 现在，让我们进入激动人心的部分——定义我们的模型！我们首先需要一个嵌入层，即将输入序列中的每个标记映射到一个向量的层。这个嵌入层可以随机初始化。我们还需要一个位置嵌入层来编码序列中的单词顺序。惯例是将这两个嵌入相加，即求和。KerasNLP 有一个 keras_nlp.layers.TokenAndPositionEmbedding层可以为我们完成上述所有步骤。
#
# 我们的 FNet 分类模型由三层组成，顶部keras_nlp.layers.FNetEncoder 有一层。keras.layers.Dense
#
# 注意：对于 FNet，屏蔽填充标记对结果的影响很小。在官方实现中，填充标记没有被屏蔽。

input_ids = keras.Input(shape=(None,), dtype="int64", name="input_ids")

x = keras_nlp.layers.TokenAndPositionEmbedding(
    vocabulary_size=VOCAB_SIZE,
    sequence_length=MAX_SEQUENCE_LENGTH,
    embedding_dim=EMBED_DIM,
    mask_zero=True,
)(input_ids)

x = keras_nlp.layers.FNetEncoder(intermediate_dim=INTERMEDIATE_DIM)(inputs=x)
x = keras_nlp.layers.FNetEncoder(intermediate_dim=INTERMEDIATE_DIM)(inputs=x)
x = keras_nlp.layers.FNetEncoder(intermediate_dim=INTERMEDIATE_DIM)(inputs=x)


x = keras.layers.GlobalAveragePooling1D()(x)
x = keras.layers.Dropout(0.1)(x)
outputs = keras.layers.Dense(1, activation="sigmoid")(x)

fnet_classifier = keras.Model(input_ids, outputs, name="fnet_classifier")


# 训练我们的模型
# 我们将使用准确性来监控验证数据的训练进度。让我们训练我们的模型 3 个 epoch。

fnet_classifier.summary()
fnet_classifier.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
fnet_classifier.fit(train_ds, epochs=EPOCHS, validation_data=val_ds)


# 计算测试精度。
fnet_classifier.evaluate(test_ds, batch_size=BATCH_SIZE)


# 与 Transformer 模型的比较
# 让我们将我们的 FNet 分类器模型与 Transformer 分类器模型进行比较。我们保持所有参数/超参数相同。例如，我们使用 TransformerEncoder三层。
# 我们将正面数量设置为 2。
#
NUM_HEADS = 2
input_ids = keras.Input(shape=(None,), dtype="int64", name="input_ids")


x = keras_nlp.layers.TokenAndPositionEmbedding(
    vocabulary_size=VOCAB_SIZE,
    sequence_length=MAX_SEQUENCE_LENGTH,
    embedding_dim=EMBED_DIM,
    mask_zero=True,
)(input_ids)

x = keras_nlp.layers.TransformerEncoder(
    intermediate_dim=INTERMEDIATE_DIM, num_heads=NUM_HEADS
)(inputs=x)
x = keras_nlp.layers.TransformerEncoder(
    intermediate_dim=INTERMEDIATE_DIM, num_heads=NUM_HEADS
)(inputs=x)
x = keras_nlp.layers.TransformerEncoder(
    intermediate_dim=INTERMEDIATE_DIM, num_heads=NUM_HEADS
)(inputs=x)


x = keras.layers.GlobalAveragePooling1D()(x)
x = keras.layers.Dropout(0.1)(x)
outputs = keras.layers.Dense(1, activation="sigmoid")(x)

transformer_classifier = keras.Model(input_ids, outputs, name="transformer_classifier")


transformer_classifier.summary()
transformer_classifier.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
transformer_classifier.fit(train_ds, epochs=EPOCHS, validation_data=val_ds)

# 我们获得了大约 94% 的训练准确度和大约 86.5% 的验证准确度。训练模型大约需要 146 秒（在 Colab 上使用 16 GB Tesla T4 GPU）。
#
# 让我们计算测试精度。

transformer_classifier.evaluate(test_ds, batch_size=BATCH_SIZE)

# [0.46172526478767395, 0.8539599776268005]


# 让我们做一个表格并比较两个模型。
# 我们可以看到，FNet 显着加快了我们的运行时间（1.7 倍），而整体准确度只牺牲了一点点（下降了 0.75%）。

#     FNet 分类器	transformer分类器
# 训练时间	86 秒	146 秒
# 训练精度	92.34%	93.85%
# 验证准确性	85.21%	86.42%
# 测试精度	83.94%	84.69%
#参数	2,321,921	2,520,065

def main():
    pass


if __name__ == '__main__':
    main()