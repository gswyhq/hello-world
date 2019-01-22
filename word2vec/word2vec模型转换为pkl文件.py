#!/usr/bin/python3
# coding=utf-8

import gensim
import pickle

from collections import OrderedDict
model = gensim.models.KeyedVectors.load_word2vec_format('./news_12g_baidubaike_20g_novel_90g_embedding_64.bin', binary=True)
word_vec = OrderedDict()
for key in model.index2word:
    word_vec[key] = model[key]

del model     #把模型给word_vec，所以Model删掉。

filepath = './word_vec.pkl'

with open(filepath, 'wb')as f:
    pickle.dump(word_vec, f)
    
# with open(filepath, 'rb')as f:
#     word2vec_model = pickle.load(f)
    
# with open('/home/gswyhq/data/model/word2vec/word_vec.pkl', 'rb')as f:
#     word2vec_model = pickle.load(f)

def main():
    pass


if __name__ == '__main__':
    main()