#----coding:utf-8----------------------------------------------------------------
# 名称:格式化gensim的语料库向量
# 目的:
#
# 作者:      gswyhq
#
# 日期:      2016-01-01
# 版本:      Python 3.3.5
# 系统:      win32
# Email:     gswyhq@126.com
#-------------------------------------------------------------------------------


from gensim import corpora

#较为常用的格式是Market Matrix 格式
# create a toy corpus of 2 documents, as a plain Python list
corpus = [[(1, 0.5)], []]  # make one document empty, for the heck of it
corpora.MmCorpus.serialize(r'F:\python\data\corpus.mm', corpus)# 存储到磁盘上，以后备用

#除此之外的格式还有：Joachim’s SVMlight format, Blei’s LDA-C format and GibbsLDA++ format.
corpora.SvmLightCorpus.serialize(r'F:\python\data\corpus.svmlight', corpus) # Joachim’s SVMlight format
corpora.BleiCorpus.serialize(r'F:\python\data\corpus.lda-c', corpus) # Blei’s LDA-C format
corpora.LowCorpus.serialize(r'F:\python\data\corpus.low', corpus) # GibbsLDA++ format.

#加载储存的语料库
corpus = corpora.MmCorpus(r'F:\python\data\corpus.mm')
print(corpus)#语料库对象是流，因此通常你将不能够直接打印出来

#打印语料库的一种方式：完全加载到内存中
print(list(corpus)) #调用列表（）将任何序列转换为纯Python列表

#另一种方法：每次打印一份文档，利用流媒体接口
for doc in corpus:
    print(doc)
#第二种方法显然是更多的内存友好的，但对于测试和开发目的，没有什么比调用列表（主体）的简单性。

#语料库与NumPy and SciPy的兼容性小
#与numpy相互转换
#corpus = gensim.matutils.Dense2Corpus(numpy_matrix)
#numpy_matrix = gensim.matutils.corpus2dense(corpus, num_terms=number_of_corpus_features)

#与scipy.sparse matrices相互转换
#corpus = gensim.matutils.Sparse2Corpus(scipy_sparse_matrix)
#scipy_csc_matrix = gensim.matutils.corpus2csc(corpus)

def main():
    pass

if __name__ == '__main__':
    main()
