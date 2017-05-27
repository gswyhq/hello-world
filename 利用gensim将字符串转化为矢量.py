#----coding:utf-8----------------------------------------------------------------
# 名称:利用gensim将字符串转化为矢量
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

documents = ["Human machine interface for lab abc computer applications",
             "A survey of user opinion of computer system response time",
             "The EPS user interface management system",
             "System and human system engineering testing of EPS",
             "Relation of user perceived response time to error measurement",
             "The generation of random binary unordered trees",
             "The intersection graph of paths in trees",
             "Graph minors IV Widths of trees and well quasi ordering",
             "Graph minors A survey"]
# 删除常用词，并标记化
stoplist = set('for a of the and to in'.split())
texts = [[word for word in document.lower().split() if word not in stoplist]
         for document in documents]

# 删除仅仅出现一次的词
from collections import defaultdict
frequency = defaultdict(int)
#这里的defaultdict(function_factory)构建的是一个类似dictionary的对象，
#其中keys的值，自行确定赋值，但是values的类型，是function_factory的类实例，
#而且具有默认值。比如default(int)则创建一个类似dictionary对象，里面任何的
#values都是int的实例，而且就算是一个不存在的key, d[key] 也有一个默认值，
#这个默认值是int()的默认值0.
for text in texts:
    for token in text:
        frequency[token] += 1

texts = [[token for token in text if frequency[token] > 1]
         for text in texts]

from pprint import pprint   # pretty-printer
pprint(texts)

dictionary = corpora.Dictionary(texts)
dictionary.save(r'F:\python\data\deerwester.dict') # 保存字典以便将来查询
#在这里，分配了一个唯一的整数ID，以显示与gensim.corpora.dictionary.Dictionary类语料库的所有单词。
print(dictionary)

#扫过文字，收集字数统计和相关统计数据。最后，我们看到有12不同词语的处理主体，
#这意味着每个文件将由12个号码来表示（即，由一个12维矢量）。要查看单词及其ID之间的映射
# 即，通过这些文档抽取一个“词袋（bag-of-words)“，将文档的token映射为id：
print(dictionary.token2id)

new_doc = "Human computer interaction"
new_vec = dictionary.doc2bow(new_doc.lower().split())
print(new_vec) # 单词"interaction"没有出现在字典中，被忽略
#函数doc2bow（）只计算每个不同的单词出现的次数，字转换为其整数字标识，并返回其
#结果作为一个稀疏矢量。稀疏向量[（0,1），（1,1）]，因此读取“Human computer interaction”,
#单词computer (id 0) and human (id 1) 各一次;其它十个字典中的字出现（隐含）零次。

#将用字符串表示的文档转换为用id表示的文档向量：
corpus = [dictionary.doc2bow(text) for text in texts]
corpora.MmCorpus.serialize(r'F:\python\data\deerwester.mm', corpus) # 存储到磁盘上，以后备用
#这句出错：File "E:\Python33\lib\site-packages\smart_open-1.3.1-py3.3.egg\smart_open\smart_open_lib.py", line 111, in smart_open
#NotImplementedError: unknown file mode wb+
#参考：http://stackoverflow.com/questions/34396300/gensim-installation-on-yosemite-using-anaconda
# pip uninstall smart_open，解决问题

print(corpus)
#corpus,是一个二维数组，内部的每一个数组代表一篇文档，内部数组是由多个2个int的元组，第一个代表单词的ID号，第二个代表单词出现的次数
#基于这些文档向量，我们可以用来训练模型


#注意上面语料库完全驻留在内存中
#若需处理大量文件，需Corpus Streaming – One Document at a Time处理。

def main():
    pass

if __name__ == '__main__':
    main()
