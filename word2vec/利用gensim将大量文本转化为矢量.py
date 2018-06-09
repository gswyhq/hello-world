#----coding:utf-8----------------------------------------------------------------
# 名称:利用gensim将大量文本转化为矢量
# 目的:
#
# 作者:      gswewf
#
# 日期:      2016-01-01
# 版本:      Python 3.3.5
# 系统:      win32
# Email:     gswewf@126.com
#-------------------------------------------------------------------------------
from gensim import corpora, models, similarities
from collections import defaultdict

class MyCorpus(object):
    def __iter__(self):
        for line in open(r'f:\python\data\mycorpus.txt'):
            # 假设每行一个文件，单词用空格隔开
            yield dictionary.doc2bow(line.lower().split())

#构建字典，而无需加载所有文本到内存中：
# 收集所有的tokens统计
dictionary = corpora.Dictionary(line.lower().split() for line in open(r'f:\python\data\mycorpus.txt'))
# 删除停用词和只出现一次的词
stoplist = set('for a of the and to in'.split())
stop_ids = [dictionary.token2id[stopword] for stopword in stoplist
            if stopword in dictionary.token2id]
once_ids = [tokenid for tokenid, docfreq in dictionary.iteritems() if docfreq == 1]
dictionary.filter_tokens(stop_ids + once_ids) # 删除停用词和只出现一次的词
dictionary.compactify() # 删除被删除的字序列后的间隙
print(dictionary)
#扫过文字，收集字数统计和相关统计数据。最后，我们看到有12不同词语的处理主体，
#这意味着每个文件将由12个号码来表示（即，由一个12维矢量）。要查看单词及其ID之间的映射
# 即，通过这些文档抽取一个“词袋（bag-of-words)“，将文档的token映射为id：
print(dictionary.token2id)

corpus_memory_friendly = MyCorpus() # 并不加载到内存中
print(corpus_memory_friendly)
#语料库现在是一个对象。我们没有定义任何的方式来打印，因此打印输出只在对象的内存地址。
#不是非常有用。要查看成分的载体，让我们遍历语料库，并打印每个文档向量（一次一个）
for vector in corpus_memory_friendly:
    print(vector)



def main():
    pass

if __name__ == '__main__':
    main()
