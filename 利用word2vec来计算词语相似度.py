#----coding:utf-8----------------------------------------------------------------
# 名称:利用word2vec来计算词语相似度
# 目的:
# http://www.52nlp.cn/%e4%b8%ad%e8%8b%b1%e6%96%87%e7%bb%b4%e5%9f%ba%e7%99%be%e7%a7%91%e8%af%ad%e6%96%99%e4%b8%8a%e7%9a%84word2vec%e5%ae%9e%e9%aa%8c
# http://rare-technologies.com/word2vec-in-python-part-two-optimizing/
# http://radimrehurek.com/gensim/models/word2vec.html
# http://radimrehurek.com/gensim/tutorial.html
# 日期:      2015-12-31
# 版本:      Python 3.3.5
# 系统:      win32
# Email:     gswyhq@126.com
#-------------------------------------------------------------------------------
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os.path
import sys
import multiprocessing

import jieba
from gensim.corpora import WikiCorpus
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

def split_word(text):
    '''利用jieba对text进行分词，返回所得分词组成的list.'''
    split_text=jieba.cut(text, cut_all=True)#全分类

    #读取样本文本
    #去除停用词，同时构造样本词的字典
    with open(r'f:\python\data\stopwords.txt',encoding='utf-8') as f:
        stop_text = f.read( )
    f_stop_seg_list=stop_text.split('\n')
    new_text=[t for t in split_text if (t not in f_stop_seg_list)and len(t)>1]
    return ' '.join(new_text)

def read_post_data():
    '''该文件提供11月卡片/帖子具体的内容，文件每行通过'\t'分隔，各列字段意义如下
    id —帖子id
    title —帖子title，可能为空
    content --帖子内容'''
    file=r'F:\python\data\data-时间黑客_数据挖掘赛数据_contest_data\post_data.txt'
    data={}
    with open(file,encoding='utf-8')as f:
        lines=f.readlines()
    for i in lines[1:]:#遍历除标题行以外的所有数据
        text=i.split('\t')
        id=text[0].strip()
        title=text[1].strip()
        content=text[2].strip()
        data[id]=title*3+content #构造一个帖子id为键，3倍权重title+一倍content为值的字典
    return data

def dict_to_text():
    data=read_post_data()
    text={}
    for i,t in data.items():
        text[i]=split_word(t)
    return text

if __name__ == '__main__':
    #输入文件：每篇文章转换位1行text文本，并且去掉了标点符号等内容
    #中文数据，与英文处理过程相似，也分两个步骤，不过需要对中文数据特殊处理一下，包括繁简转换，中文分词，去除非utf-8字符等。
    inp=r'f:\python\data\jiebatext.txt'#输入文件
    outp1=r"F:\python\data\text.model"#gensim中默认格式的word2vec model
    outp2 =r"F:\python\data\text.vector"#原始c版本word2vec的vector格式的模型

    model = Word2Vec(LineSentence(inp), size=400, window=5, min_count=5,
            workers=multiprocessing.cpu_count())
#    参数解释：
#1.sg=1是skip-gram算法，对低频词敏感；默认sg=0为CBOW算法。
#2.size是输出词向量的维数，值太小会导致词映射因为冲突而影响结果，值太大则会耗内存并使算法计算变慢，一般值取为100到200之间。
#3.window是句子中当前词与目标词之间的最大距离，3表示在目标词前看3-b个词，后面看b个词（b在0-3之间随机）。
#4.min_count是对词进行过滤，频率小于min-count的单词则会被忽视，默认值为5。
#5.negative和sample可根据训练结果进行微调，sample表示更高频率的词被随机下采样到所设置的阈值，默认值为1e-3。
#6.hs=1表示层级softmax将会被使用，默认hs=0且negative不为0，则负采样将会被选择使用。
#7.workers控制训练的并行，此参数只有在安装了Cpython后才有效，否则只能使用单核。

#word2vec的两种形式：CBOW和Skip-gram模型
#CBOW去除了上下文各词的词序信息，使用上下文各词的平均值。
#skip-gram和CBOW正好相反，它使用单一的焦点词作为输入，经过训练然后输出它的目标上下文

    # trim unneeded model memory = use(much) less RAM
    #model.init_sims(replace=True)
    model.save(outp1)
    model.save_word2vec_format(outp2, binary=False)

    #加载测试这个模型
    #model = gensim.models.Word2Vec.load_word2vec_format(outp2, binary=False)
    #mo=model.most_similar("the")


