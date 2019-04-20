#/usr/lib/python3.5
# -*- coding: utf-8 -*-

#http://ltp.readthedocs.org/zh_CN/latest/faq.html
#https://github.com/HIT-SCIR/pyltp/blob/master/example/example.py
#http://www.ltp-cloud.com/demo/
#各个字段的意思：http://www.ltp-cloud.com/intro/


import sys, os

ROOTDIR = os.path.join(os.path.dirname(__file__), os.pardir)
sys.path.append(os.path.join(ROOTDIR, "lib"))

# Set your own model path
MODELDIR=os.path.join('/usr/local', "ltp_data")

from pyltp import Segmentor, Postagger, Parser, NamedEntityRecognizer, SementicRoleLabeller

def split_words(sentence = "中国进出口银行与中国银行加强合作",type_list=0):
    """分词,若type_list=True,则返回以列表返回分词后的结果。"""
    segmentor = Segmentor()
    segmentor.load(os.path.join(MODELDIR, "cws.model"))
    words = segmentor.segment(sentence)
    if type_list:
        return [i for i in words]
    return words

def words_cixing(words=["中国","进出口","银行","与","中国银行","加强","合作"],type_list=0,pos=0):
    """词性标注,若type_list=True,则返回以列表返回标注词性后的结果。
    词性标记集：LTP中采用863词性标注集
    词性说明见：http://www.ltp-cloud.com/intro/
    若type_list为真，则返回['ns', 'v', 'n', 'c', 'ni', 'v', 'v']
    若pos为真，则返回['中国/ns', '进出口/v', '银行/n', '与/c', '中国银行/ni', '加强/v', '合作/v']
    默认返回是生成器列表
    """
    if type(words)==str:
        words=split_words(words)
    postagger = Postagger()
    postagger.load(os.path.join(MODELDIR, "pos.model"))
    postags = postagger.postag(words)
    # list-of-string parameter is support in 0.1.5
    # postags = postagger.postag(["中国","进出口","银行","与","中国银行","加强","合作"])
    if type_list :
        return [i for i in postags]
    if pos:
        return ['{}/{}'.format(k,v)for k,v in zip(words,[i for i in postags])]
    return postags

def jufa_fenxi(words,postags):
    """句法分析"""
    parser = Parser()
    parser.load(os.path.join(MODELDIR, "parser.model"))
    arcs = parser.parse(words, postags)

    print ("\t".join("%d:%s" % (arc.head, arc.relation) for arc in arcs))

def mingming_shiti(words,postags):
    """命名实体。机构名(Ni)人名(Nh)地名(Ns)"""
    recognizer = NamedEntityRecognizer()
    recognizer.load(os.path.join(MODELDIR, "ner.model"))
    netags = recognizer.recognize(words, postags)
    print ("\t".join(netags))

def yuyijuese(words, postags, netags, arcs):
    """语义角色标注  """
    labeller = SementicRoleLabeller()
    labeller.load(os.path.join(MODELDIR, "srl/"))
    roles = labeller.label(words, postags, netags, arcs)

    for role in roles:
        print (role.index, "".join(
                ["%s:(%d,%d)" % (arg.name, arg.range.start, arg.range.end) for arg in role.arguments]))

segmentor.release()
postagger.release()
parser.release()
recognizer.release()
labeller.release()


'''
Created on 2015-4-29

@author: 郭喜跃

import sys, os
import json
from pyltp import Segmentor, Postagger, Parser, NamedEntityRecognizer, SementicRoleLabeller
ROOTDIR = os.path.join(os.path.dirname(__file__), os.pardir)
sys.path.append(os.path.join(ROOTDIR, "lib"))
# 设置模型文件的路径
MODELDIR=os.path.join(ROOTDIR, "ltp_data")

def callLTP(sentence):# 参数就是待处理的句子
    # sentence = "国家主席胡锦涛携夫人刘永青出访俄罗斯。"

    #分词功能
    segmentor = Segmentor()
    segmentor.load(os.path.join(MODELDIR, "cws.model"))
    words = segmentor.segment(sentence)
    #print ("\t".join(words))

    #词性标注功能
    postagger = Postagger()
    postagger.load(os.path.join(MODELDIR, "pos.model"))
    postags = postagger.postag(words)
    #print ("\t".join(postags))

    #句法依存关系
    parser = Parser()
    parser.load(os.path.join(MODELDIR, "parser.model"))
    arcs = parser.parse(words, postags)
    #print ("\t".join("%d:%s" % (arc.head, arc.relation) for arc in arcs))

    #实体识别
    recognizer = NamedEntityRecognizer()
    recognizer.load(os.path.join(MODELDIR, "ner.model"))
    netags = recognizer.recognize(words, postags)
    #print ("\t".join(netags))

    #语义角色标注，这个我没有用到，所以全部注释了
    #labeller = SementicRoleLabeller()
    #labeller.load(os.path.join(MODELDIR, "srl/"))
    #roles = labeller.label(words, postags, netags, arcs)
    #for role in roles:
    #    print (role.index, "".join(["%s:(%d,%d)" % (arg.name, arg.range.start, arg.range.end) for arg in role.arguments]))

    # 结果整合为json。这是重点。我把原代码的print全部注释。
    resultJson=[] #创建一个空列表，用于保存json数据。
    for index in range(len(words)):#遍历结果
        resultJson.append({'id':index,'cont':words[index],'pos':postags[index],'relate':arcs[index].relation,'ne':netags[index]}) #将各功能的结果对应地添加到json中

    return resultJson # 返回函数结果

'''