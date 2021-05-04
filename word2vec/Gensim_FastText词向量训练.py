#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from gensim.models import FastText

sentences = [["你", "是", "谁"], ["我", "是", "中国人"]]

model = FastText(sentences,  size=4, window=3, min_count=1, iter=10,min_n = 3 , max_n = 6,word_ngrams = 0)
model['你']  # 词向量获得的方式
model.wv['你'] # 词向量获得的方式

# 模型保存与加载
model.save(fname)
model = FastText.load(fname)

# 那么既然gensim之中的fasttext,那么也有这么一种方式：

fasttext_model.wv.save_word2vec_format('temp/test_fasttext.txt', binary=False)
fasttext_model.wv.save_word2vec_format('temp/test_fasttext.bin', binary=True)

def main():
    pass


if __name__ == '__main__':
    main()