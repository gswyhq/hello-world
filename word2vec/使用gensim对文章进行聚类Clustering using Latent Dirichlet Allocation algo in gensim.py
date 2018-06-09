#----coding:utf-8----------------------------------------------------------------
# 名称:使用gensim对文章进行聚类Clustering using Latent Dirichlet Allocation algo in gensim
# 目的:Document Clustering with Python
# http://brandonrose.org/clustering
# https://gist.github.com/balamuru/4727614
# 作者:      gswewf
#
# 日期:      2016-01-02
# 版本:      Python 3.3.5
# 系统:      win32
# Email:     gswewf@126.com
#-------------------------------------------------------------------------------

import logging
import os
import gensim
from gensim.models import TfidfModel, LsiModel


test_data_dir  = r"F:\python\data\reuters21578\txt"

#日志系统的基本配置。
#format:使用指定的格式字符串处理函数。
#level:设置记录器为指定级别
#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def iter_documents(top_directory):
    """遍历所有文件，生成文档 (=list of utf8 tokens)"""
    for root, dirs, files in os.walk(top_directory):
        #filter(function or None, sequence) -> list, tuple, or string
        #function是一个谓词函数，接受一个参数，返回布尔值True或False。
        #filter函数会对序列参数sequence中的每个元素调用function函数，最后返回的结果包含调用结果为True的元素。
        #返回值的类型和参数sequence的类型相同
        for file in filter(lambda file: file.endswith('.txt'), files):
            #print file
            document = open(os.path.join(root, file)).read() # 读取整个文档
            yield gensim.utils.tokenize(document, lower=True) # 将documents分词

class MyCorpus(object):
    def __init__(self, top_dir):
        self.top_dir = top_dir
        self.dictionary = gensim.corpora.Dictionary(iter_documents(top_dir))#词频统计，返回（word_id，字频）
        self.dictionary.filter_extremes(no_below=1, no_above=0.5,keep_n=30000)
        # 滤过小于`no_below`（绝对数） 的词或者大于`no_above`（整个语料库的相对数），
        #并只保留前keep_n个词，若keep_n=None,则全部保留
        # **注意**：由于缩小差距（即设置了keep_n不为None时），同一个词可以前后调用此函数后都有不同的词ID！修剪后，收缩导致的词ID的差距

    def __iter__(self):
        #如果一个类想被用于for ... in循环，类似list或tuple那样，就必须实现一个__iter__()方法，
        #该方法返回一个迭代对象，然后，Python的for循环就会不断调用该迭代对象
        #的__next__()方法拿到循环的下一个值，直到遇到StopIteration错误时退出循环。
        for tokens in iter_documents(self.top_dir):
            yield self.dictionary.doc2bow(tokens)
            # 将文档(a list of words) 转换成 bag-of-words format = list of `(token_id, token_count)` 2-tuples.
            # 可通过参数allow_update来设置对模型的更新或只读

corpus = MyCorpus(test_data_dir) # 创建一个字典

for vector in corpus: # 每个文档转换成 a bag-of-word vector后的输出
    print (vector)
    break

print ("创建模型")
tfidf_model = TfidfModel(corpus)#转换成局部/全局加权TF_IDF矩阵，它可以将一个简单的计数表示成TFIDF空间。
# tfidf = TfidfModel(corpus)
# print(tfidf[some_doc])#输出模型
# tfidf.save('/tmp/foo.tfidf_model')#保存模型

lsi_model = LsiModel(corpus)
#LSA(latent semantic analysis)潜在语义分析，也被称为LSI(latent semantic index)，
#是一种新的索引和检索方法。该方法和传统向量空间模型(vector space model)一样使用向量来表示词(terms)和文档(documents)，
#并通过向量间的关系(如夹角)来判断词及文档间的关系；而不同的是，LSA将词和文档映射到潜在语义空间。
#同义词和多义词如何导致传统向量空间模型检索精确度的下降。
#LSA潜在语义分析的目的，就是要找出词(terms)在文档和查询中真正的含义，也就是潜在语义，从而解决上节所描述的问题。

topic_id = 0
for topic in lsi_model.show_topics():
    topic_id+=1
    print ("TOPIC (LSI) " + str(topic_id) + " : ", topic)

print('#'*50)
print(lsi_model.num_topics)
for i in range(0, lsi_model.num_topics-1):
    if lsi_model.print_topic(i):
        print (lsi_model.print_topic(i))

corpus_tfidf = tfidf_model[corpus]
corpus_lsi = lsi_model[corpus]

lsi_model_2 = LsiModel(corpus_tfidf, id2word=corpus.dictionary, num_topics=300)
corpus_lsi_2 = lsi_model_2[corpus]
print ('完成创建模型')


print('*'*10, lsi_model_2 .print_topics(5))

topic_id = 0
for topic in lsi_model_2.show_topics():
    print ("TOPIC (LSI2) " , str(topic_id) , " : " , topic)
    group_topic = [doc for doc in corpus_lsi_2 if doc[topic_id][1] > 0.5]
    print (str(group_topic))
    topic_id+=1





print ("文档加工 " + str(lsi_model_2.docs_processed))

for doc in corpus_lsi_2: # 无论 bow->tfidf 还是 tfidf->lsi 实际上是在此运行
    print ("Doc " + str(doc))


#模型的保存
#corpus.dictionary.save("dictionary.dump")
#
#tfidf_model.save("model_tfidf.dump")
#corpus_tfidf.save("corpus_tfidf.dump")
#
#lsi_model.save("model_lsi.dump")
#corpus_lsi.save("corpus_lsidump")
#
#
#lsi_model_2.save("model_lsi_2.dump")
#corpus_lsi_2.save("corpus_lsi_2.dump")

for doc in corpus_tfidf:
    print (doc)
