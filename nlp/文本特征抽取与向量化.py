# -*- coding: utf-8 -*-
#文本特征抽取与向量化
#来源：http://blog.csdn.net/lsldd/article/details/41520953

import scipy as sp
import numpy as np
from sklearn.datasets import load_files
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import  TfidfVectorizer

'''加载数据集，切分数据集80%训练，20%测试
Python的sklearn.datasets支持从目录读取所有分类好的文本。
不过目录必须按照一个文件夹一个标签名的规则放好。
比如本文使用的数据集共有2个标签，一个为“net”，一个为“pos”，
每个目录下面有多个文本文件。
'''
#数据来自康奈尔大学：http://www.cs.cornell.edu/people/pabo/movie-review-data/
#加载数据返回一个字典对象
movie_reviews = load_files(r'F:\python\data\txt_sentoken')

#将数组或矩阵，拆分成用于随机训练和测试的子集
doc_terms_train, doc_terms_test, y_train, y_test\
    = train_test_split(movie_reviews.data, movie_reviews.target, test_size = 0.3)

'''BOOL型特征下的向量空间模型，注意，测试样本调用的是transform接口

有些单词对情感分类是毫无意义的。比如上述数据中的“of”，“I”之类的单词。
这类词有个名字，叫“Stop_Word“（停用词）。这类词是可以完全忽略掉不做统计的。
显然忽略掉这些词，词频记录的存储空间能够得到优化，而且构建速度也更快。

把每个单词的词频作为重要的特征也存在一个问题。比如上述数据中的”movie“，
在12个样本中出现了5次，但是出现正反两边次数差不多，没有什么区分度
。而”worth“出现了2次，但却只出现在pos类中，显然更具有强烈的刚晴色彩，即区分度很高。
因此，我们需要引入TF-IDF（Term Frequency-Inverse Document Frequency，
词频和逆向文件频率）对每个单词做进一步考量。
TF（词频）的计算很简单，就是针对一个文件t，某个单词Nt 出现在该文档中的频率。
比如文档“I love this movie”，单词“love”的TF为1/4。如果去掉停用词“I"和"this"，则为1/2。
IDF（逆向文件频率）的意义是，对于某个单词t，凡是出现了该单词的文档数Dt，
占了全部测试文档D的比例，再求自然对数。
比如单词“movie“一共出现了5次，而文档总数为12，因此IDF为ln(5/12)。
很显然，IDF是为了凸显那种出现的少，但是占有强烈感情色彩的词语
。比如“movie”这样的词的IDF=ln(12/5)=0.88，远小于“love”的IDF=ln(12/1)=2.48。
TF-IDF就是把二者简单的乘在一起即可。这样，求出每个文档中，
每个单词的TF-IDF，就是我们提取得到的文本特征值。
'''

#stop_words = 'english'，表示使用默认的英文停用词。
#可以使用count_vec.get_stop_words()查看TfidfVectorizer内置的所有停用词。
#词频的计算使用的是sklearn的TfidfVectorizer。这个类继承于CountVectorizer，
#在后者基本的词频统计基础上增加了如TF-IDF之类的功能。
#count_vec构造时默认传递了max_df=1，因此TF-IDF都做了规格化处理，以便将所有值约束在[0,1]之间。
#数据集可能存在非法字符问题。传入了decode_error = 'ignore'，以忽略这些非法字符。
count_vec = TfidfVectorizer(binary = False, decode_error = 'ignore',\
                            stop_words = 'english')
x_train = count_vec.fit_transform(doc_terms_train)
#count_vec.fit_transform的结果是一个巨大的矩阵。
x_test  = count_vec.transform(doc_terms_test)
x       = count_vec.transform(movie_reviews.data)
y       = movie_reviews.target
print(doc_terms_train)

#特征数组
print('特征',count_vec.get_feature_names())

#查看文档字符串
print("文档字符串",x_train.toarray())

#分类标签
print('分类标签',movie_reviews.target)