#!/usr/bin/python3
# coding: utf-8

from scipy.sparse.csr import csr_matrix

# TfidfVectorizer中的fit_transform方法就是利用scipy的稀疏矩阵构建并返回term-document矩阵：

docs = [['中华', '民族', '血汗'], ['人民', '血汗'], ['共和', '国', '数学', '中华', '英语', '物理']]

indptr = [0]  # 存放的是行偏移量
indices = []  # 存放的是data中元素对应的列编号（列编号可重复）
data = []  # 存放的是非0数据元素
vocabulary = {}  # key是word词汇，value是列编号
for d in docs:  # 遍历每个文档
    for term in d:  # 遍历文档的每个词汇term
        # setdefault如果term不存在，则将新term和他的列
        # 编号len(vocabulary)加入到词典中，返回他的编号；
        # 如果term存在，则不填加，返回已存在的编号
        index = vocabulary.setdefault(term, len(vocabulary))
        indices.append(index)
        data.append(1)
    indptr.append(len(indices))
# csr_matrix可以将同一个词汇次数求和
csr_mat = csr_matrix((data, indices, indptr), dtype=int).toarray()

print(csr_mat)