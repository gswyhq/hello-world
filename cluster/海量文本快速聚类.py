#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 事先不知道文本数据聚类数的情况下,针对大规模的文本聚类，可以考虑单遍聚类(Single-Pass Clustering)
# Single-pass聚类算法同时是一种增量聚类算法（Incremental Clustering Algorithm），每个文档只需要流过算法一次，所以被称为single-pass，
# 效率远高于K-means或KNN等算法。它可以很好的应用于话题监测与追踪、在线事件监测等社交媒体大数据领域，特别适合流式数据（Streaming Data），
# 比如微博的帖子信息，因此适合对实时性要求较高的文本聚类场景。
#
# 处理步骤
# Single-pass算法顺序处理文本，以第一篇文档为种子，建立一个新主题。
# 之后再进行新进入文档与已有主题的相似度，将该文档加入到与它相似度最大的且大于一定阈值的主题中。
# 如果与所有已有话题相似度都小于阈值，则以该文档为聚类种子，建立新的主题类别。
#
# 其算法流程如下：
# （1）以第一篇文档为种子，建立一个主题；
# （2）基于词袋模型将文档X向量化；
# （3）将文档X与已有的所有话题均做相似度计算，可采用欧氏距离、余弦距离等
# （4）找出与文档X具有最大相似度的已有主题；
# （5）若相似度值大于阈值theta，则把文档X加入到有最大相似度的主题中，跳转至 7；
# （6）若相似度值小于阈值theta， 则文档X不属于任一已有主题， 需创建新的主题类别，同时将当前文本归属到新创建的主题类别中；
# （7）聚类结束，等待下一篇文档进入 。

# single pass聚类也是存在弱点的，主要是：
# （1）依赖数据读入的顺序；
# （2）阈值设定较为困难；
# （3）单独效果使用较差。

import sys, os
from jieba import posseg
from jieba.analyse import textrank, tfidf

class Document():

    def __init__(self, doc_id, content, features):
        self.doc_id = doc_id
        self.features = features  # 文本的特征，这里是分词结果
        self.content = content  # 原文


class Cluster():

    def __init__(self, cluster_id, center_doc_id):
        self.cluster_id = cluster_id  # 簇的id，用来从map中获取这个簇的信息
        self.center_doc_id = center_doc_id  # 核心文档的id，用来从map红获取这个文档的信息。为了减少文档信息的备份数量，簇里只存储这个
        self.members = [center_doc_id]  # 簇成员的id列表。由于只遍历一遍(这是single-pass的核心竞争力之一)，不存在重复的可能，这里使用list

    def add_doc(self, doc_id):
        self.members.append(doc_id)

# 增加倒排索引
class SinglePassV2():

    def __init__(self):
        self.document_map = {}  # 存储文档信息，id-content结构。当然value也可以使用对象存储文档的其他信息。
        self.cluster_map = {}  # 存储簇的信息，id-cluster_object结构。
        self.cluster_iindex = {}  # word-cluster_ids结构

    # 提取文本特征
    def get_words(self, text):
        # words = HanLP.segment(text)
        # words = tfidf(text)
        words = textrank(text, allowPOS=('ns', 'n', 'vn', 'v', 'nr', 'nz'))
        words = list("".join(words))
        # words = list(map(str, words))
        return words

    # 输入文档列表，进行聚类。现实中，我们遇到的文档会带有id等信息，这里为了简单，只有文本内容，所以需要生成id,一遍存取。
    def fit(self, document_list):
        # 对文档进行预处理
        self.preprocess(document_list)
        self.clutering()

    def similar(self, cluster, document):
        cluster_feature = set(self.document_map[cluster.center_doc_id].features)
        document_feature = set(document.features)
        #         print(cluster_feature, document_feature)
        similarity = len(cluster_feature & document_feature) / len(cluster_feature | document_feature)
        if similarity > 0.2:
            return True
        else:
            return False

    # 对所有文档分词，并生成id
    def preprocess(self, document_list):
        for i in range(len(document_list)):
            doc_id = "document_" + str(i)
            content = document_list[i]
            words = self.get_key_words(content)
            document = Document(doc_id, content, words)
            self.document_map[doc_id] = document

    # 提取文本特征。这里使用文档内频次最高的K个词语。实际应用中可以用TF-IDF
    def get_key_words(self, text, K=5):
        words = self.get_words(text)
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:K]
        keywords = list(map(lambda x: x[0], keywords))
        return keywords

    def clutering(self):
        for doc_id in self.document_map:
            #             print(doc_id, self.document_map[doc_id])
            words = self.document_map[doc_id].features
            if_特立独行 = True
            for cluster_id in self.get_cand_clusters(words):
                cluster = self.cluster_map[cluster_id]
                if self.similar(cluster, self.document_map[doc_id]):
                    cluster.add_doc(doc_id)
                    if_特立独行 = False
                    break
            if if_特立独行:
                new_cluser_id = "cluster_" + str(len(self.cluster_map))
                print(new_cluser_id)
                new_cluster = Cluster(new_cluser_id, doc_id)
                self.cluster_map[new_cluser_id] = new_cluster

                for word in self.document_map[new_cluster.center_doc_id].features:
                    if word not in self.cluster_iindex: self.cluster_iindex[word] = []
                    self.cluster_iindex[word].append(new_cluser_id)

    def get_cand_clusters(self, words):
        cand_cluster_ids = []
        for word in words:
            cand_cluster_ids.extend(self.cluster_iindex.get(word, []))
        return cand_cluster_ids

    # 打印所有簇的简要内容
    def show_clusters(self):
        for cluster_id in self.cluster_map:
            cluster = self.cluster_map[cluster_id]
            print(cluster.cluster_id, cluster.center_doc_id, cluster.members, [self.document_map[doc_id].content for doc_id in cluster.members])

def main():
    docs = ["我爱北京天安门，天安门上太阳升。",
            "我要开着火车去北京，看天安门升旗。",
            "我们的家乡，在希望的田野上。",
            "我的老家是一片充满希望的田野。"]
    single_passor = SinglePassV2()
    single_passor.fit(docs)
    single_passor.show_clusters()


if __name__ == '__main__':
    main()

# 也可以参考：https://github.com/liuhuanyong/SinglepassTextCluster
