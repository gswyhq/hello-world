#----coding:utf-8----------------------------------------------------------------
# 名称:利用gensim进行文章的相似查询
# 目的:
# 来源：http://radimrehurek.com/gensim/tut3.html
# 作者:      gswewf
#
# 日期:      2016-01-01
# 版本:      Python 3.3.5
# 系统:      win32
# Email:     gswewf@126.com
#-------------------------------------------------------------------------------
from gensim import corpora, models, similarities

dictionary = corpora.Dictionary.load(r'F:\python\data\deerwester.dict')
corpus = corpora.MmCorpus(r'F:\python\data\deerwester.mm')
print(corpus)

#先用这个小语料库定义一个2维LSI空间：
lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=2)

#定义被查询的词
doc = "Human computer interaction"
vec_bow = dictionary.doc2bow(doc.lower().split())#将被查询词向量化
vec_lsi = lsi[vec_bow] # 用之前训练好的LSI模型将其映射到二维的topic空间
print(vec_lsi)

#初始化查询结构
index = similarities.MatrixSimilarity(lsi[corpus]) # 变换空间到LSI 并建立索引
#similarities.MatrixSimilarity只有在适当的时候全组向量能够装入内存。例如，一百万个文档语料库需要2GB的RAM中的256维的LSI空间

#若没有2GB内存，你需要使用similarities.Similarity类。此类工作在固定的内存中，
#通过分割为多个文件索引在磁盘上，称为碎片。它采用similarities.MatrixSimilarity和similarities.SparseMatrixSimilarity

#索引的保存与加载
index.save(r'F:\python\data\deerwester.index')
index = similarities.MatrixSimilarity.load(r'F:\python\data\deerwester.index')

sims = index[vec_lsi] # 对语料库进行相似性查询
print(list(enumerate(sims))) # 输出(document_number, document_similarity)
#余弦测量返回的相似的范围<-1，1>（越大，越相似），使得第一文件具有0.99809301一个得分等

#按相似性降序排列，输出
sims = sorted(enumerate(sims), key=lambda item: -item[1])
print(sims) # 输出 sorted (document number, similarity score) 2-tuples


def main():
    pass

if __name__ == '__main__':
    main()
