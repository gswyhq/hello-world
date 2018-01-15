#!/usr/bin/python
# -*- coding:utf8 -*- #
from __future__ import generators
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import sys, os

if sys.version >= '3':
    PY3 = True
else:
    PY3 = False

if PY3:
    import pickle
    import configparser
else:
    import cPickle as pickle
    from codecs import open
    import ConfigParser as configparser

import time, os, random,json
import re
import logging
from pypinyin import lazy_pinyin, load_phrases_dict, TONE2, load_single_dict

PATH=sys.path[0]
#stream: 指定将日志的输出流，可以指定输出到sys.stderr, sys.stdout或者文件，默认输出到sys.stderr，当stream和filename同时指定时，stream被忽略
logging.basicConfig(level=logging.DEBUG, 
                format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s', 
                datefmt='%a,  %d %b %Y %H:%M:%S', 
                stream=sys.stdout, 
                #filename=os.path.join(sys.path[0], 'log', '{}日志.log'.format(time.strftime("%Y%m%d"))),  #%H%M
                filemode='a+')
if not os.path.isdir(os.path.join(sys.path[0], 'log', )):
    os.mkdir(os.path.join(sys.path[0], 'log', ))
    
def read_btxt(outfile):
    '''读取成语文件，生成相应的拼音字典,共44482词，结果如下所示：
    {"钝口拙腮": ["dun", "kou", "zhuo", "sai"], 
    "怜我怜卿": ["lian", "wo", "lian", "qing"]}
    版本1，共四万多词
    '''
    infile=os.path.join(PATH,'成语接龙语料（4字）.txt')
    dict_data={}
    
    with open(infile,'r',encoding='utf8')as f:
        data=f.readlines()
        for d in data:
            line=d.strip() #.decode('gb18030')
            if len(line)!=4:
                continue
            dict_data.setdefault(line,lazy_pinyin(line))
    with open(outfile,'w',encoding='utf8')as fo:
        json.dump(dict_data,fo,ensure_ascii=0) 
    print('总共成语数：',len(dict_data.keys()))     
    
#
def read_btxt2(infile,outfile):
    '''读取成语文件，生成相应的拼音字典,共44482词，结果如下所示：
    {"钝口拙腮": ["dun", "kou", "zhuo", "sai"], 
    "怜我怜卿": ["lian", "wo", "lian", "qing"]}
    版本2，共55222词
    '''
    dict_data={}
    
    with open(infile,'r',encoding='utf8')as f:
        data=json.load(f)
        for d in data:
            line=d.strip() #.decode('gb18030')
            if len(line)!=4:
                continue
            dict_data.setdefault(line,lazy_pinyin(line))
    with open(outfile,'w',encoding='utf8')as fo:
        json.dump(dict_data,fo,ensure_ascii=0) 
    print('总共成语数：',len(dict_data.keys()))     
    
def main():
    infile='/home/gswyhq/idiom/成语/成语.json'
    outfile='/home/gswyhq/idiom/idiom.json'
    read_btxt2(infile,outfile)

if __name__ == "__main__":
    main()
