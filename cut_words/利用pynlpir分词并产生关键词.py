#/usr/lib/python3.5
# -*- coding: utf-8 -*-
#pynlpir
import os,sys
import pynlpir


def read_stop_words():
    #读取停用词
    #tf = open("stop_words.txt", "r")
    tf=open('/home/gswyhq/问答问题/问答系统/stop_words.txt')
    lines = tf.readlines()
    tf.close()
    stop_words = [line[:-1] for line in lines]
    stop_words.append("是")
    return stop_words
    
def find_wh_word_pos(line):
    ''' Wh words指疑问词
    查找疑问词出现的位置
    find w/h words in line,
    return (pos, wh-word)
    '''
    wh_words=['什么', '谁', '哪', '何时', '第几', '多少', '几', '为什么']
    for wh in wh_words:
        
        for word_pos in range(len(line)):
            word = line[word_pos]
            if word.startswith(wh):
                return (word_pos, word.split("/")[0])
    return (-1, "")

def gen_cent_word(line):
    ''' 产生一个中心词，
    产生方法：1，疑问词之后的第一个名词,若没有则采用方法2
    方法2：若方法1没有，且‘是’在句子中出现了，这是就返回‘是’前面最近的一个名词。若没有则返回空值
    generate central word
    in a question sentence
    '''
    l = len(line)
    pos = find_wh_word_pos(line)[0]
    #find_wh_word_pos函数，查找经过分词的问题的疑问词在句子中出现的位置及出现的疑问词
    #返回值如：(4, '第几'),(7, '什么')
    if pos != -1:
        for i in range(pos, l):
            #从疑问词之后开始查找
            attr = line[i].split("/")[1] #获取词性
            if len(attr)>0 and attr[0]=='n': #如果是名词则返回
                return line[i]
    if "是/v" in line:
        pos = 0
        while (line[pos]!="是/v"):
            pos += 1
        for i in range(pos, -1, -1):
            attr = line[i].split("/")[1]
            if attr[0]=='n':
                return line[i]
    return ""

def gen_keywords(line,pos=1):
    ''' 通过问题，产生关键词
    关键词产生过程是：
    1、将中心词添加到关键词表中，
    2、将除中心词之外的名词添加到关键词表中，
    3、将不在停用词表中的动词添加到关键词表中
    4，最后将其他的词，添加到关键词表中
    总结：产生关键词即是将line里的词，按重要程度排序，越重要的排在list的前，不重要的排在list的后，返回值是一个同等长度的list
    generate keywords for a question
    return a list of keywords
    '''
    # ['莱昂纳多/nrt', '第一次/m', '获得/v', '奥斯卡/nr', '提名/v', '是/v', '多少岁/m']
    l = len(line)

    # 中心词central word
    cent_word = gen_cent_word(line)
    if cent_word:
        ret = [cent_word]#把中心词添加到ret
    else:
        ret=[]

    # 名词nouns
    nouns = []
    for i in range(l):
        #print ("#%s#" % (line[i]))
        attr = line[i].split("/")[1]
        if len(attr)>0 and attr[0]=='n' and line[i]!=cent_word:
            nouns.append(line[i])
    ret += nouns #把除中心词之外的名词也添加到ret中

    # 动词verbs
    verbs = []
    stop_words=read_stop_words()
    for i in range(l):
        word = line[i].split("/")[0]
        attr = line[i].split("/")[1]
        #if len(attr) > 0 and attr[0]=='v' and not (line[i] in stop_words):
        if len(attr) > 0 and attr[0]=='v' and not (word in stop_words):
            verbs.append(line[i])
    ret += verbs #添加不在停用词词表中的动词

    # other words
    for i in range(l):
        word = line[i].split("/")[0]
        if not line[i] in ret:
            ret.append(line[i])

    if pos:
        return ret
    else:
        return [t.split('/')[0] for t in ret]

def words_cixing(question,pos=1):
    #pos=1，标注词性；否则不标注
    pynlpir.open()
    if pos:
        pos1=['{}/{}'.format(k,v)for k,v in pynlpir.segment(question, pos_names=None,pos_tagging=pos)]
    else:
        pos0=pynlpir.segment(question)
    pynlpir.close()
    if pos:
        return pos1
    else :
        return pos0
        
def main(question):
    #keywords=['刘德华','歌手']
    postags=words_cixing(question)#分词，并词性标注
    #print(postags)
    ret=gen_keywords(postags,pos=0)#产生关键词
    return ret
    
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print ("请输入问题,或者问题文件！")
    else:
        print ("开始产生关键词 ...%s" % sys.argv[1])
        ret=main(sys.argv[1])
        print(ret)
