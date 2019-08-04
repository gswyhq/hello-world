#!/usr/bin/python3
# coding: utf-8

# a.导入相关库
import math
import pickle
import numpy as np
import jieba
import functools
from gensim import corpora,models

# https://mp.weixin.qq.com/s/6ZnvyfEwqqrPYzBt6ER0-g

# CORPUS_FILE_PATH = '/home/gswyhq/data/ChnSentiCorp_htl_all.csv' # 2.8M
CORPUS_FILE_PATH = '/home/gswyhq/data/zhwiki/extracted/AA/wiki_01' # 286M
MODEL_PATH = '/home/gswyhq/Downloads/topic_model.pkl'
# LDA的训练就是根据现有的数据集生成 文档-主题分布矩阵 和 主题-词分布矩阵。
# P(词 | 文档)=P（词 | 主题）P（主题 | 文档）
#
# 训练一个关键词提取算法需要以下步骤：
# 加载已有的文档数据集
# 加载停用词表
# 对数据集中的文档进行分词
# 根据停用词表，过滤干扰词
# 根据训练集训练算法

# b.定义好停用词表的加载方法


def get_stopword_list():
    stop_word_path = '/home/gswyhq/data/stopwords.txt'
    stopword_list = [sw.replace('', '') for sw in open(stop_word_path).readlines()]
    return stopword_list


# c.定义一个分词方法


def seg_to_list(sentence, pos=False):
    seg_list = jieba.lcut(sentence)
    return seg_list


# d.定义干扰词过滤方法：根据分词结果对干扰词进行过滤


def word_filter(seg_list, pos=False):
    stopword_list = get_stopword_list()
    filter_list = [str(word) for word in seg_list if not word in stopword_list and len(word) > 1]
    return filter_list


# e.加载数据集，对数据集中的数据分词和过滤干扰词，每个文本最后变成一个非干扰词组成的词语列表

def load_data(pos=False):
    doc_list = []
    ll = []
    for line in open(CORPUS_FILE_PATH, 'r', encoding='utf-8'):
        line = line.strip()
        if line:
            ll.append(line)
    content =' '.join(ll)
    seg_list = seg_to_list(content, pos)
    filter_list = word_filter(seg_list, pos)
    doc_list.append(filter_list)
    return doc_list

# f.训练LDA模型

# 余弦相似度计算
def calsim(l1, l2):
    a, b, c = 0.0, 0.0, 0.0
    for t1, t2 in zip(l1, l2):
        x1 = t1[1]
        x2 = t2[1]
        a += x1 * x1
        b += x1 * x1
        c += x2 * x2
    sim = a / math.sqrt(b * c) if not (b * c) == 0 else 0.0
    return sim

# doc_list：加载数据集方法的返回结果
# keyword_num：关键词数量
# model：主题模型的具体算法
# num_topics：主题模型的主题数量
class TopicModel(object):
    def __init__(self, doc_list, keyword_num, model='LDA', num_topics=4):
        # 使用gensim的接口，将文本转换为向量化的表示
        self.dictionary = corpora.Dictionary(doc_list)
        # 使用BOW模型向量化
        corpus = [self.dictionary.doc2bow(doc) for doc in doc_list]
        # 对每个词，根据TF-IDF进行加权，得到加权后的向量表示
        self.tfidf_model = models.TfidfModel(corpus)
        self.corpus_tfidf = self.tfidf_model[corpus]
        self.keyword_num = keyword_num
        self.num_topics = num_topics
        self.model = self.train_lda()

        # 得到数据集的 主题-词分布
        word_dic = self.word_dictionary(doc_list)
        self.wordtopic_dic = self.get_wordtopic(word_dic)

    def train_lda(self):
        lda = models.LdaModel(self.corpus_tfidf, num_topics=self.num_topics, id2word=self.dictionary)
        return lda

    def get_wordtopic(self, word_dic):
        wordtopic_dic = {}
        for word in word_dic:
            single_list = [word]
            wordcorpus = self.tfidf_model[self.dictionary.doc2bow(single_list)]
            wordtopic = self.model[wordcorpus]
            wordtopic_dic[word] = wordtopic
        return wordtopic_dic

    def get_simword(self, word_list):
        sentcorpus = self.tfidf_model[self.dictionary.doc2bow(word_list)]
        senttopic = self.model[sentcorpus]
        sim_dic = {}
        for k, v in self.wordtopic_dic.items():
            if k not in word_list:
                continue
            sim = calsim(v, senttopic)
            sim_dic[k] = sim
        print("sim_dic: {}".format(sim_dic))
        # for k, v in sorted(sim_dic.items(), key=functools.cmp_to_key(cmp), reverse=True)[:self.keyword_num]:
        #     print(k + "/", end='')
        # print()
        # 词空间构建方法和向量化方法，在没有gensim接口时的一般处理方法

    def word_dictionary(self, doc_list):
        dictionary = []
        for doc in doc_list:
            dictionary.extend(doc)
        dictionary = list(set(dictionary))
        return dictionary

    def doc2bowvec(self, word_list):
        vec_list = [1 if word in word_list else 0 for word in self.dictionary]
        return vec_list


# g.调用主函数，对目标文本进行关键词提取
def main(IS_TRAIN=True):
    text = '会上,中华社会救助基金会与“第二届中国爱心城市大会”承办方晋江市签约,许嘉璐理事长接受晋江市参与“百万孤老关爱行动”向国家重点扶贫地区捐赠的价值400万元的款物。'
    pos = False
    seg_list = seg_to_list(text, pos)
    filter_list = word_filter(seg_list, pos)
    print('LDA模型结果: {}'.format(filter_list))
    # topic_extract(filter_list, 'LDA', pos)
    if IS_TRAIN:
        doc_list = load_data()
        keyword_num = 10
        topic = TopicModel(doc_list, keyword_num, model='LDA', num_topics=4)
        with open(MODEL_PATH, 'wb')as f:
            pickle.dump(topic, f)
    else:
        with open(MODEL_PATH, 'rb')as f:
            topic = pickle.load(f, encoding='iso-8859-1')
    topic.get_simword(filter_list)
    print(topic.doc2bowvec(filter_list))


if __name__ == '__main__':
    main(IS_TRAIN=False)