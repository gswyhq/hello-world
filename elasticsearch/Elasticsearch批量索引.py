#!/usr/bin/python
# -*- coding:utf8 -*- #
from __future__ import  generators
from __future__ import  division
from __future__ import  print_function
from __future__ import  unicode_literals
import sys,os,json

if sys.version >= '3':
    PY3 = True
else:
    PY3 = False

if PY3:
    import pickle
else:
    import cPickle as pickle
    from codecs import open

索引操作
对于单条索引，可以调用create或index方法。
from datetime import datetime
from elasticsearch import Elasticsearch
es = Elasticsearch() #create a localhost server connection, or Elasticsearch("ip")
es.create(index="test-index", doc_type="test-type", id=1,
  body={"any":"data", "timestamp": datetime.now()})

Elasticsearch批量索引的命令是bulk，目前Python API的文档示例较少，花了不少时间阅读源代码才弄清楚批量索引的提交格式。

from datetime import datetime
from elasticsearch import Elasticsearch
from elasticsearch import helpers
es = Elasticsearch("10.18.13.3")
j = 0
count = int(df[0].count())
actions = []
while (j < count):
   action = {
        "_index": "tickets-index",
        "_type": "tickets",
        "_id": j + 1,
        "_source": {
              "crawaldate":df[0][j],
              "flight":df[1][j],
              "price":float(df[2][j]),
              "discount":float(df[3][j]),
              "date":df[4][j],
              "takeoff":df[5][j],
              "land":df[6][j],
              "source":df[7][j],
              "timestamp": datetime.now()}
        }
  actions.append(action)
  j += 1

  if (len(actions) == 500000):
    helpers.bulk(es, actions)
    del actions[0:len(actions)]

if (len(actions) > 0):
  helpers.bulk(es, actions)
  del actions[0:len(actions)]
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  from datetime import datetime  
  
from elasticsearch import Elasticsearch  
from elasticsearch import helpers  
  
es = Elasticsearch()  
  
actions = []  
  
f=open('index.txt')  
i=1  
for line in f:  
    line = line.strip().split(' ')  
    action={  
        "_index":"image",  
        "_type":"imagetable",  
        "_id":i,  
        "_source":{  
                u"图片名":line[0].decode('utf8'),  
                u"来源":line[1].decode('utf8'),  
                u"权威性":line[2].decode('utf8'),  
                u"大小":line[3].decode('utf8'),  
                u"质量":line[4].decode('utf8'),  
                u"类别":line[5].decode('utf8'),  
                u"型号":line[6].decode('utf8'),  
                u"国别":line[7].decode('utf8'),  
                u"采集人":line[8].decode('utf8'),  
                u"所属部门":line[9].decode('utf8'),  
                u"关键词":line[10].decode('utf8'),  
                u"访问权限":line[11].decode('utf8')      
            }  
        }  
    i+=1  
    actions.append(action)  
    if(len(actions)==500):  
        helpers.bulk(es, actions)  
        del actions[0:len(actions)]  
  
if (len(actions) > 0):  
    helpers.bulk(es, actions)  
      
   

每句话的含义还是很明显的，这里需要说几点，首先是index.txt是以utf8编码的，所以需要decode('utf8')转换成unicode对象，并且“图片名”前需要加u，否则ES会报错
导入的速度还是很快的，2000多条记录每秒


  
def main():
    pass


if __name__ == "__main__":
    main()
