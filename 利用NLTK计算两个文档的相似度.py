#----coding:utf-8----------------------------------------------------------------
# 名称:利用NLTK计算两个文档的相似度
# 目的:
# 参考:http://www.52nlp.cn/%e5%a6%82%e4%bd%95%e8%ae%a1%e7%ae%97%e4%b8%a4%e4%b8%aa%e6%96%87%e6%a1%a3%e7%9a%84%e7%9b%b8%e4%bc%bc%e5%ba%a6%e4%b8%89
# 作者:      gswyhq
#
# 日期:      2016-01-14
# 版本:      Python 3.3.5
# 系统:      win32
# Email:     gswyhq@126.com
#-------------------------------------------------------------------------------
import nltk

#加载数据
courses = [line.strip() for line in open(r'f:\python\data\coursera_corpus',encoding='utf-8')]
courses_name = [course.split('\t')[0] for course in courses]
print( courses_name[0:10])

#将单词小写化
texts_lower = [[word for word in document.lower().split()] for document in courses]
print (texts_lower[0])

#其中很多标点符号和单词是没有分离的，所以我们引入nltk的word_tokenize函数，并处理相应的数据
from nltk.tokenize import word_tokenize
texts_tokenized = [[word.lower() for word in word_tokenize(document)] for document in courses]
print (texts_tokenized[0])

#对课程的英文数据进行tokenize之后，我们需要去停用词，幸好NLTK提供了一份英文停用词数据：

from nltk.corpus import stopwords
english_stopwords = stopwords.words('english')
print(english_stopwords)

#过滤课程语料中的停用词：
texts_filtered_stopwords = [[word for word in document if not word in english_stopwords] for document in texts_tokenized]
print (texts_filtered_stopwords[0])

#停用词被过滤了，不过发现标点符号还在，这个好办，我们首先定义一个标点符号list:
english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%']

#然后过滤这些标点符号：
texts_filtered = [[word for word in document if not word in english_punctuations] for document in texts_filtered_stopwords]
print (texts_filtered[0])

#对这些英文单词词干化（Stemming)，NLTK提供了好几个相关工具接口可供选择，
#具体参考这个页面: http://nltk.org/api/nltk.stem.html , 可选的工具包括Lancaster Stemmer, Porter Stemmer等知名的英文Stemmer。
from nltk.stem.lancaster import LancasterStemmer
st = LancasterStemmer()
texts_stemmed = [[st.stem(word) for word in docment] for docment in texts_filtered]
print (texts_stemmed[0])

#去掉在整个语料库中出现次数为1的低频词
all_stems = sum(texts_stemmed, [])
stems_once = set(stem for stem in set(all_stems) if all_stems.count(stem) == 1)
texts = [[stem for stem in text if stem not in stems_once] for text in texts_stemmed]

#引入gensim，并快速的做课程相似度的实验
from gensim import corpora, models, similarities
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

dictionary = corpora.Dictionary(texts)

corpus = [dictionary.doc2bow(text) for text in texts]

tfidf = models.TfidfModel(corpus)

corpus_tfidf = tfidf[corpus]

#训练topic数量为10的LSI模型：
lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=10)

index = similarities.MatrixSimilarity(lsi[corpus])

#基于LSI模型的课程索引建立完毕，我们以Andrew Ng教授的机器学习公开课为例，这门课程在我们的coursera_corpus文件的第211行，也就是：

print (courses_name[210])#Machine Learning

#现在我们就可以通过lsi模型将这门课程映射到10个topic主题模型空间上，然后和其他课程计算相似度：
ml_course = texts[210]
ml_bow = dictionary.doc2bow(ml_course)
ml_lsi = lsi[ml_bow]
print( ml_lsi)

sims = index[ml_lsi]
sort_sims = sorted(enumerate(sims), key=lambda item: -item[1])

#取按相似度排序的前10门课程：
print( sort_sims[0:10])

for i,n in sort_sims[0:10]:
    print ('相似的课程依次是：',courses_name[i])

def main():
    pass

if __name__ == '__main__':
    main()
