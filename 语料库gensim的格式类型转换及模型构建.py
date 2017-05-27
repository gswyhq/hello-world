#----coding:utf-8----------------------------------------------------------------
# 名称:语料库gensim的格式类型转换及模型构建
# 目的:
#
# 作者:      gswyhq
#
# 日期:      2016-01-01
# 版本:      Python 3.3.5
# 系统:      win32
# Email:     gswyhq@126.com
#-------------------------------------------------------------------------------
from gensim import corpora, models, similarities
from pprint import pprint

#从一个向量表示转换文档转换成另一种。这个过程有两个目标：
#为了显示隐藏的结构，在语料，探索词与词之间的关系，并利用它们来描述文档中一个新的（希望）更语义的方式。
#为了使文档表示更加紧凑。这既提高了效率（新表示消耗更少的资源）和有效性（边际数据的趋势将被忽略，降噪）。
dictionary = corpora.Dictionary.load(r'F:\python\data\deerwester.dict')
corpus = corpora.MmCorpus(r'F:\python\data\deerwester.mm')
print(corpus)

#创建一个转换类型
#基于“训练文档”计算一个TF-IDF“模型”
tfidf = models.TfidfModel(corpus) # 第一步，初始化模型
doc_bow = [(0, 1), (1, 1)]

#基于这个TF-IDF模型，我们可以将上述用词频表示文档向量表示为一个用tf-idf值表示的文档向量：
print(tfidf[doc_bow]) # 第二步，使用模型转换载体

#或者转换到一个corpus
corpus_tfidf = tfidf[corpus]
for doc in corpus_tfidf:
    print(doc)

#打印一下tfidf模型中的信息
#单词ID及其所在的总文档数
print (tfidf.dfs)

#单词ID及其分数
print (tfidf.idfs)

#转换也可以是链式的，在一个转换后，继续转换
#有了tf-idf值表示的文档向量，我们就可以训练一个LSI模型
lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=2) # 初始化LSI转换，设置num_topics=2，创建一个二维的
#将训练文档向量组成的矩阵SVD分解，并做了一个秩为2（num_topics=2）的近似SVD分解，
#有了这个lsi模型，我们就可以将文档映射到一个二维的topic空间中

corpus_lsi = lsi[corpus_tfidf] # 在原始语料创建一个双重封装: bow->tfidf->fold-in-lsi
pprint(lsi.print_topics(2))

for doc in corpus_lsi: # both bow->tfidf and tfidf->lsi transformations are actually executed here, on the fly
    print(doc)

#转换后的模型保存与加载
lsi.save(r'F:\python\data\model.lsi') # same for tfidf, lda, ...
lsi = models.LsiModel.load(r'F:\python\data\model.lsi')

#几个流行的向量空间模型算法

#TFIDF 公式是向量空间模型中应用比较成功的计算特征项权值的方法。
#研究发现,该公式忽略了特征项在文本集的分布比例和离散程度这两个影响特征项对文本表示贡献度 的重要因素。
model = tfidfmodel.TfidfModel(bow_corpus, normalize=True)

#LSI训练的特别之处在于我们能继续“训练”在任何时候，只需通过提供更多的培训文件。
#Bradford. 2008. An empirical study of required dimensionality for large-scale latent semantic indexing applications.
model = lsimodel.LsiModel(tfidf_corpus, id2word=dictionary, num_topics=300)#纬度为200-500是比较常用的

model.add_documents(another_tfidf_corpus) # LSI 完成训练 tfidf_corpus + another_tfidf_corpus
lsi_vec = model[tfidf_vec] # 将一些新文件加入 LSI 空间，而不影响模型

model.add_documents(more_documents) # tfidf_corpus + another_tfidf_corpus + more_documents
lsi_vec = model[tfidf_vec]

#随机投影，RP的目标是减少向量空间维数。这是一个非常有效的（包括内存和CPU的友好的）
#的方式来近似文档之间TFIDF距离，通过投掷在一个小的随机性。建议的目标维度又是在数百/千，这取决于你的数据集。
model = rpmodel.RpModel(tfidf_corpus, num_topics=500)


#隐含狄利克雷分布，LDA又是从另一个转变袋的字计数到低维的主题空间。
#LDA是LSA的概率扩展（也称为多项式，PCA），所以LDA的主题可以被解释为通过词语概率分布。
#这些分布是，就像LSA，从训练语料库自动推断。文件将逐一解释为对这些主题的（软）的混合物（再次，就像LSA）。
#Hoffman, Blei, Bach. 2010. Online learning for Latent Dirichlet Allocation.
model = ldamodel.LdaModel(bow_corpus, id2word=dictionary, num_topics=100)

#HDP是一个非参数贝叶斯方法
#Wang, Paisley, Blei. 2011. Online variational inference for the hierarchical Dirichlet process
model = hdpmodel.HdpModel(bow_corpus, id2word=dictionary)

def main():
    pass

if __name__ == '__main__':
    main()
