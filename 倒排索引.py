#! /usr/lib/python3
# -*- coding: utf-8 -*-

'''
倒排索引（英语：Inverted index），也常被称为反向索引、置入档案或反向档案，是一种索引方法，被用来存储在全文搜索下某个单词在一个文档或者一组文档中的存储位置的映射。它是文档检索系统中最常用的数据结构。

http://www.2cto.com/kf/201304/203046.html


倒排索引分析：


以英文为例，下面是要被索引的文本：
T0 = "it is what it is"
T1 = "what is it"
T2 = "it is a banana"
对相同的文字，我们得到后面这些完全反向索引，有文档数量和当前查询的单词结果组成的的成对数据。 同样，文档数量和当前查询的单词结果都从零开始。所以，"banana": {(2, 3)} 就是说 "banana"在第三个文档里(T2)，而且在第三个文档的位置是第四个单词(地址为 3)。
"a":        {(2, 2)}
"banana": {(2, 3)}
"is":       {(0, 1), (0, 4), (1, 1), (2, 1)}
"it":       {(0, 0), (0, 3), (1, 2), (2, 0)}
"what":   {(0, 2), (1, 0)}
如果我们执行短语搜索"what is it" 我们得到这个短语的全部单词各自的结果所在文档为文档0和文档1。但是这个短语检索的连续的条件仅仅在文档1得到。



倒排索引的Python实现：

'''
#-------------------------------------------------------------------------------
# Name:        InvertedIndex
# Purpose:     倒排索引
# Created:     02/04/2013
# Copyright:   (c) neng 2013
# Licence:     <your licence>
#-------------------------------------------------------------------------------
import re
import string
#对199801.txt进行处理, 去除词性标注以、日期及一些杂质. (保留段落结构)
#输入:199801.txt
#输出:199801_new.txt
def pre_file(filename):
    print("读取语料库文件%r....."%filename)
    src_data = open(filename).read()
    #去除词性标注、‘19980101-01-001-001’、杂质如‘[’、‘]nt’
    des_data =re.compile(r'(\/\w+)|(\d+\-\S+)|(\[)|(\]\S+)').sub('',src_data)
    des_filename = "199801_new.txt"
    print("正在写入文件%r....."%des_filename)
    open(des_filename,'w').writelines(des_data)
    print("处理完成!")


#建立倒排索引
#输入:199801_new.txt
#输出:my_index.txt  格式(从0开始): word (段落号,段落中位置) ..
def create_inverted_index(filename):
    print("读取文件%r....."%filename)
    src_data = open(filename).read()
    #变量说明
    sub_list = [] #所有词的list,  用来查找去重等
    word = []     #词表文件
    result = {}   #输出结果 {单词:index}

    #建立词表
    sp_data = src_data.split()
    set_data = set(sp_data) #去重复
    word = list(set_data) #set转换成list, 否则不能索引

    src_list = src_data.split("\n") #分割成单段话vv
    #建立索引
    for w in range(0,len(word)):
        index = []  #记录段落及段落位置 [(段落号,位置),(段落号,位置)...]
        for i in range(0,len(src_list)):  #遍历所有段落
            #print(src_list[i])
            sub_list = src_list[i].split()
            #print(sub_list)
            for j in range(0,len(sub_list)):  #遍历段落中所有单词
                #print(sub_list[j])
                if sub_list[j] == word[w]:
                    index.append((i,j))
            result[word[w]] = index

    des_filename = "my_index.txt"
    print("正在写入文件%r....."%des_filename)
    #print(result)
    #print(word)
    #print(len(word))
    writefile = open(des_filename,'w')
    for k in result.keys():
        writefile.writelines(str(k)+str(result[k])+"\n")
    print("处理完成!")

#主函数
def main():
    #pre_file("199801.txt") #初步处理语料库, 得到199801_new.txt
    #根据199801_new.txt建立索引, 得到myindex.txt(由于199801_new.txt太大, 建立索引时间太长, 因此只截取一部分放入199801_test.txt)
    create_inverted_index("199801_test.txt")

#运行
if __name__ == '__main__':
    main()

#-------------------------------------------------------------------------------
# Name:        InvertedIndex
# Purpose:     倒排索引
# Created:     02/04/2013
# Copyright:   (c) neng 2013
# Licence:     <your licence>
#-------------------------------------------------------------------------------
import re
import string
#对199801.txt进行处理, 去除词性标注以、日期及一些杂质. (保留段落结构)
#输入:199801.txt
#输出:199801_new.txt
def pre_file(filename):
    print("读取语料库文件%r....."%filename)
    src_data = open(filename).read()
    #去除词性标注、‘19980101-01-001-001’、杂质如‘[’、‘]nt’
    des_data =re.compile(r'(\/\w+)|(\d+\-\S+)|(\[)|(\]\S+)').sub('',src_data)
    des_filename = "199801_new.txt"
    print("正在写入文件%r....."%des_filename)
    open(des_filename,'w').writelines(des_data)
    print("处理完成!")


#建立倒排索引
#输入:199801_new.txt
#输出:my_index.txt  格式(从0开始): word (段落号,段落中位置) ..
def create_inverted_index(filename):
    print("读取文件%r....."%filename)
    src_data = open(filename).read()
    #变量说明
    sub_list = [] #所有词的list,  用来查找去重等
    word = []     #词表文件
    result = {}   #输出结果 {单词:index}

    #建立词表
    sp_data = src_data.split()
    set_data = set(sp_data) #去重复
    word = list(set_data) #set转换成list, 否则不能索引

    src_list = src_data.split("\n") #分割成单段话vv
    #建立索引
    for w in range(0,len(word)):
        index = []  #记录段落及段落位置 [(段落号,位置),(段落号,位置)...]
        for i in range(0,len(src_list)):  #遍历所有段落
            #print(src_list[i])
            sub_list = src_list[i].split()
            #print(sub_list)
            for j in range(0,len(sub_list)):  #遍历段落中所有单词
                #print(sub_list[j])
                if sub_list[j] == word[w]:
                    index.append((i,j))
            result[word[w]] = index

    des_filename = "my_index.txt"
    print("正在写入文件%r....."%des_filename)
    #print(result)
    #print(word)
    #print(len(word))
    writefile = open(des_filename,'w')
    for k in result.keys():
        writefile.writelines(str(k)+str(result[k])+"\n")
    print("处理完成!")

#主函数
def main():
    #pre_file("199801.txt") #初步处理语料库, 得到199801_new.txt
    #根据199801_new.txt建立索引, 得到myindex.txt(由于199801_new.txt太大, 建立索引时间太长, 因此只截取一部分放入199801_test.txt)
    create_inverted_index("199801_test.txt")

#运行
if __name__ == '__main__':
    main()

'''

主要算法说明：

输入: 待建立索引的文件（199801_test.txt）

输出: 索引文件（my_index.txt）

建立倒排索引的算法采用全文搜索，然后记录其段落号与在段落中的位置。

'''