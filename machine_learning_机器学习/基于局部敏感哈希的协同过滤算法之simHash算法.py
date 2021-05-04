#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
simhash算法分为5个步骤：分词、hash、加权、合并、降维
百度的去重算法最简单，就是直接找出此文章的最长的n句话，做一遍hash签名。n一般取3。
具体simhash步骤如下：
（1）将文档分词，取一个文章的TF-IDF权重最高的前20个词（feature）和权重（weight）。即一篇文档得到一个长度为20的（feature：weight）的集合。
（2）对其中的词（feature），进行普通的哈希之后得到一个64为的二进制，得到长度为20的（hash : weight）的集合。
（3）根据（2）中得到一串二进制数（hash）中相应位置是1是0，对相应位置取正值weight和负值weight。
例如一个词进过（2）得到（010111：5）进过步骤（3）之后可以得到列表[-5,5,-5,5,5,5]，即对一个文档，我们可以得到20个长度为64的列表[weight，-weight...weight]。
（4）对（3）中20个列表进行列向累加得到一个列表。如[-5,5,-5,5,5,5]、[-3,-3,-3,3,-3,3]、[1,-1,-1,1,1,1]进行列向累加得到[-7，1，-9，9，3，9]，这样，我们对一个文档得到，一个长度为64的列表。
（5）对（4）中得到的列表中每个值进行判断，当为负值的时候去0，正值取1。例如，[-7，1，-9，9，3，9]得到010111，这样，我们就得到一个文档的simhash值了。
（6）计算相似性。连个simhash取异或，看其中1的个数是否超过3。超过3则判定为不相似，小于等于3则判定为相似。

对于64位的待查询文本的simhash code来说，如何在海量的样本库（>1M）中查询与其海明距离在3以内的记录呢？
假设我们要寻找海明距离3以内的数值，根据抽屉原理，只要我们将整个64位的二进制串划分为4块，无论如何，匹配的两个simhash code之间至少有一块区域是完全相同的，
由于我们无法事先得知完全相同的是哪一块区域，因此我们必须采用存储多份table的方式。
在本例的情况下，我们需要存储4份table，并将64位的simhash code等分成4份；
对于每一个输入的code，我们通过精确匹配的方式，查找前16位相同的记录作为候选记录

让我们来总结一下上述算法的实质：
1、将64位的二进制串等分成四块
2、调整上述64位二进制，将任意一块作为前16位，总共有四种组合，生成四份table
3、采用精确匹配的方式查找前16位
4、如果样本库中存有2^34（差不多10亿）的哈希指纹，则每个table返回2^(34-16)=262144个候选结果，大大减少了海明距离的计算成本

'''

import jieba
import jieba.analyse
import numpy as np

# 获取字符串对应的hash值
class SimhashStr():
    def __init__(self, str):
        self.str = str

    # 得到输入字符串的hash值
    def get_hash(self):
        # 取前20个关键词
        keyword = jieba.analyse.extract_tags(self.str, topK=20, withWeight=True, allowPOS=())
        keyList = []
        # 获取每个词的权重
        for feature, weight in keyword:
            # 每个关键词的权重*总单词数
            weight = int(weight * 20)
            # 获取每个关键词的特征
            feature = self.string_hash(feature)
            temp = []
            # 获取每个关键词的权重
            for i in feature:
                if i == '1':
                    temp.append(weight)
                else:
                    temp.append(-weight)
                keyList.append(temp)
        # 将每个关键词的权重变成一维矩阵
        list1 = np.sum(np.array(keyList), axis=0)
        # 获取simhash值
        simhash = ''
        for i in list1:
            # 对特征标准化表示
            if i > 0:
                simhash = simhash + '1'
            else:
                simhash = simhash + '0'
        return simhash

    def string_hash(self, feature):
        if feature == "":
            return 0
        else:
            # 将字符转为二进制，并向左移动7位
            x = ord(feature[0]) << 7
            m = 1000003
            mask = 2 ** 128 - 1
            # 拼接每个关键词中字符的特征
            for c in feature:
                x = ((x * m) ^ ord(c)) & mask
            x ^= len(feature)
            if x == -1:
                x = -2
            # 获取关键词的64位表示
            x = bin(x).replace('0b', '').zfill(64)[-64:]
            return str(x)


# 比较两个字符串的相似度
class simliary():
    def __init__(self, sim1, sim2):
        self.sim1 = sim1
        self.sim2 = sim2

    # 比较两个simhash值的相似度
    def com_sim(self):
        # 转为二进制结构
        t1 = '0b' + self.sim1
        t2 = '0b' + self.sim2
        n = int(t1, 2) ^ int(t2, 2)
        # 相当于对每一位进行异或操作
        i = 0
        while n:
            n &= (n - 1)
            i += 1
        return i

#比较大量文本中数据之间的相似度
class com_file_data_sim():
    def __init__(self, path):
        self.path = path

    # 获取文件中的数据列表
    def get_file_data(self):
        content_txt = []
        with open(self.path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                content = line.strip()
                content_txt.append(content)
        return content_txt

    # 对列表中的数据进行hash值比对
    def com_data_sim(self):
        content_data = self.get_file_data()
        content_data_hash = [SimhashStr(str1).get_hash() for str1 in content_data]
        for i in range(len(content_data) - 1):
            for y in range(i + 1, len(content_data)):
                sim1 = content_data_hash[i]
                sim2 = content_data_hash[y]
                sim = simliary(sim1, sim2).com_sim()
                print('{}, {} simhash值为: {}'.format(i, y, sim))

def main():
    com_file_data_sim('com.txt').com_data_sim()

if __name__ == '__main__':
    main()