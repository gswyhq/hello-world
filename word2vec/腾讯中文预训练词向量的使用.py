#!/usr/bin/python3
# coding: utf-8

# 词向量下载：
# wget -c -t 0 https://ai.tencent.com/ailab/nlp/data/Tencent_AILab_ChineseEmbedding.tar.gz
# 预训练的嵌入在Tencent_AILab_ChineseEmbedding.txt中。第一行显示嵌入的总数及其尺寸大小，以空格分隔。
# 在下面的每一行中，第一列表示中文单词或短语，后跟空格及其嵌入。对于每次嵌入，其在不同维度中的值由空格分隔。

# 腾讯AI Lab嵌入词汇表中有停用词（例如“的”和“是”），数字和标点符号（例如“，”和“。”）

from gensim.models import KeyedVectors
wv_from_text = KeyedVectors.load_word2vec_format(file, binary=False)

def main():
    pass


if __name__ == '__main__':
    main()
