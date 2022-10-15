#----coding:utf-8----------------------------------------------------------------
# 名称: 利用NLTK进行文本分析
# 目的:
# 参考:http://miguelmalvarez.com/2015/03/20/classifying-reuters-21578-collection-with-python-representing-the-data/
# http://www.52nlp.cn/%e5%a6%82%e4%bd%95%e8%ae%a1%e7%ae%97%e4%b8%a4%e4%b8%aa%e6%96%87%e6%a1%a3%e7%9a%84%e7%9b%b8%e4%bc%bc%e5%ba%a6%e4%b8%89
# 作者:      gswewf
#
# 日期:      2016-01-10
# 版本:      Python 3.3.5
# 系统:      win32
# Email:     gswewf@126.com
#-------------------------------------------------------------------------------
from nltk.corpus import reuters

#统计信息
def collection_stats():
	# 文档列表
	documents = reuters.fileids()
	print(len(documents),"篇文档");

	train_docs = list(filter(lambda doc: doc.startswith("train"),
                        documents));
	print(str(len(train_docs)) + "篇训练文档");

	test_docs = list(filter(lambda doc: doc.startswith("test"),
                       documents));
	print(str(len(test_docs)) + "篇测试文档");

	# 类别列表
	categories = reuters.categories();
	print(str(len(categories)) + "个类别");

	# 一个类别中的文档
	category_docs = reuters.fileids("acq");

	# 文档单词
	document_id = category_docs[0]
	document_words = reuters.words(category_docs[0]);
	print("文档单词：\n",document_words);

	# 原始文档
	print("原始文档：\n",reuters.raw(document_id));

from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

cachedStopWords = stopwords.words("english")

def tokenize(text):
	min_length = 3
	#word_tokenize：将文档转换成单词列表；lower：单词小写化；
	words = map(lambda word: word.lower(), word_tokenize(text));

	# 去除停用词
	words = [word for word in words
                  if word not in cachedStopWords]

	#单词词干化
	tokens =(list(map(lambda token: PorterStemmer().stem(token),
                  words)));
	p = re.compile('[a-zA-Z]+');

    #单词的首位是字母，并且单词长度≥3
    #re.match ：只从字符串的开始与正则表达式匹配，匹配成功返回matchobject，否则返回none；
	filtered_tokens =list(filter(lambda token:
                  p.match(token) and len(token)>=min_length,
         tokens));
	return filtered_tokens

# Return the representer, without transforming
def tf_idf(docs):
	tfidf = TfidfVectorizer(tokenizer=tokenize, min_df=3,
                        max_df=0.90, max_features=3000,
                        use_idf=True, sublinear_tf=True,
                        norm='l2');
	tfidf.fit(docs);
	return tfidf;


def feature_values(doc, representer):
	doc_representation = representer.transform([doc])
	features = representer.get_feature_names()
	return [(features[index], doc_representation[0, index])
                 for index in doc_representation.nonzero()[1]]


def main():
	train_docs = []
	test_docs = []

	for doc_id in reuters.fileids():
		if doc_id.startswith("train"):
			train_docs.append(reuters.raw(doc_id))
		else:
			test_docs.append(reuters.raw(doc_id))

	representer = tf_idf(train_docs);

	for doc in test_docs:
		print(feature_values(doc, representer))


if __name__ == '__main__':
    main()
