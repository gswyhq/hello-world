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

HOST = '172.26.1.128'
PORT = '9200'
from datetime import datetime
from elasticsearch import Elasticsearch

#连接elasticsearch,默认是9200
es = Elasticsearch(host=HOST, port=PORT)
#es = Elasticsearch(host="10.18.13.3",port=9200)
# host": "localhost", "port": 9200

# 按id搜索
result = es.get(index='musicinfo', doc_type='music', id=207409)['_source']

# 查找标题，会把标题切词后再去搜索
res = es . search (
                    index = 'musicinfo' ,
                    doc_type = 'music' ,
                    body = {
                        'query' :
                            {'match' :
                                 { 'title' : '山楂花' }},
                        "size": 1, #设置返回结果的个数
                    }
                    )

# 对title为“梨花又开放”进行搜索，不会对“梨花又开放”进行分词处理，返回title中含有“梨花又开放”的结果
res = es . search (
                    index = 'musicinfo' ,
                    doc_type = 'music' ,
                    body = {
                        'query' :
                            {'match' :
                                 { 'title' : {"query":"梨花又开放",
                                               "operator":"and"} }}
                    }
                    )

# 精确匹配歌名含有：山楂花
res = es . search (
                    index = 'musicinfo' ,
                    doc_type = 'music' ,
                    body = {
                        'query' : #query类型改成过滤（Filters） 要比查询快很多，因为和查询相比它们不需要执行打分过程, 尤其是当设置缓存过滤结果之后.
                            {'match' :
                                 { 'title' : {"query":"山楂花",
                                               "operator":"and"} }}
                    }
                    )
def test(body={ 'query' :  {'match' :   { 'title' : {"query":"山楂花",  "operator":"and"} }}}):
    import time
    start = time.time()
    res = es . search (
                    index = 'musicinfo' ,
                    doc_type = 'music' ,
                    body = body

                    )
    print("耗时：",time.time()-start)
    print(res)

res = es . search (
                    index = 'musicinfo' ,
                    doc_type = 'music' ,
                    body = {
                        "query": {
                            "multi_match": {
                              "query": "不提",
                              "type": "phrase", #type指定为phrase
                              "slop": 0,        #slop指定每个相邻词之间允许相隔多远。此处设置为0，以实现完全匹配。实际上并不是完全匹配，而是匹配到包含这个字符串
                              "fields": [
                                "title"
                              ],
                              #"analyzer": "charSplit", #分析器指定为charSplit
                              "max_expansions": 1
                            }
                          },
                    },
                    )

# 下面这样并不能匹配到结果，但数据库里头存在
res = es . search (
                    index = 'musicinfo' ,
                    doc_type = 'music' ,
                    body = {
                        'query' :
                            {'term' :
                                 { 'title' : '山楂花' }}
                    }
                    )

# 按歌手搜索
result = es.search(
    index='musicinfo',
    doc_type='music',
    body={
        "query": {
            "bool": {
                "must": [
                    {
                    "term": {"singer": "范宗沛"}
                    }
                ],
                "must_not": [ ],
                "should": [ ]
                }
            },
        "from": 0,
        "size": 10,
        "sort": [ ],
        "aggs": { }
        }
)

# 按歌名搜索
result = es.search(
    index='musicinfo',
    doc_type='music',
    body={
        "query": {
            "bool": {
                "must": [
                {
                "term": {
                "title": "花"
                }
                }
                ],
            "must_not": [ ],
            "should": [ ]
            }
        },
        "from": 0,
        "size": 10,
        "sort": [ ],
        "aggs": { }
    }
)

# 随机搜索
curl -XGET '172.26.1.128:9200/_search' -d '{
  "query": {
    "function_score" : {
      "query" : { "match_all": {} },
      "random_score" : {}
    }
  }
}';

#创建索引，索引的名字是my-index,如果已经存在了，就返回个400，
#这个索引可以现在创建，也可以在后面插入数据的时候再临时创建
es.indices.create(index='my-index',ignore)
#{u'acknowledged':True}
 
 
#插入数据,(这里省略插入其他两条数据，后面用)
es.index(index="my-index",doc_type="test-type",id=01,body={"any":"data01","timestamp":datetime.now()})
#{u'_type':u'test-type',u'created':True,u'_shards':{u'successful':1,u'failed':0,u'total':2},u'_version':1,u'_index':u'my-index',u'_id':u'1}
#也可以，在插入数据的时候再创建索引test-index
es.index(index="test-index",doc_type="test-type",id=42,body={"any":"data","timestamp":datetime.now()})
 
 
#查询数据，两种get and search
#get获取
res = es.get(index="my-index", doc_type="test-type", id=01)
print(res)
#{u'_type': u'test-type', u'_source': {u'timestamp': u'2016-01-20T10:53:36.997000', u'any': u'data01'}, u'_index': u'my-index', u'_version': 1, u'found': True, u'_id': u'1'}
print(res['_source'])
#{u'timestamp': u'2016-01-20T10:53:36.997000', u'any': u'data01'}
 
#search获取
res = es.search(index="test-index", body={"query":{"match_all":{}}})  #获取所有数据
print(res)
#{u'hits':
#    {
#    u'hits': [
#        {u'_score': 1.0, u'_type': u'test-type', u'_id': u'2', u'_source': {u'timestamp': u'2016-01-20T10:53:58.562000', u'any': u'data02'}, u'_index': u'my-index'},
#        {u'_score': 1.0, u'_type': u'test-type', u'_id': u'1', u'_source': {u'timestamp': u'2016-01-20T10:53:36.997000', u'any': u'data01'}, u'_index': u'my-index'},
#        {u'_score': 1.0, u'_type': u'test-type', u'_id': u'3', u'_source': {u'timestamp': u'2016-01-20T11:09:19.403000', u'any': u'data033'}, u'_index': u'my-index'}
#    ],
#    u'total': 5,
#    u'max_score': 1.0
#    },
#u'_shards': {u'successful': 5, u'failed': 0, u'total':5},
#u'took': 1,
#u'timed_out': False
#}
for hit in res['hits']['hits']:
    print(hit["_source"])
res = es.search(index="test-index", body={'query':{'match':{'any':'data'}}}) #获取any=data的所有值
print(res)


class exportEsData():
    """从Elasticsearch中导出数据"""

    def __init__(self, url='http://172.26.1.128:9200',index=ES_INDEX,type=ES_DOC_TYPE):
        self.url = os.path.join(url, index, type, "_search")
        self.index = index
        self.type = type
        self.size = 10000

    def exportData(self, out_file):
        print("export data 开始...")
        begin = time.time()
        # try:
        #     os.remove(out_file) # 用来删除一个文件
        # except:
        #     os.mknod() # 创建空文件
        msg = urllib2.urlopen(self.url).read()
        # print(msg)
        if not PY3 and isinstance(msg, str):
            msg = msg.decode('utf8')
        obj = json.loads(msg)
        num = obj["hits"]["total"]
        start = 0
        end = math.ceil(num/self.size) # 向上取整
        # end =  int(num/self.size)+1
        while(start<end):
            # msg = urllib2.urlopen(self.url+"?from="+str(start*self.size)+"&size="+str(self.size)).read()
            url = "{}?from={}&size={}".format(self.url,start*self.size, self.size)
            url = "{}?pretty&search_type=scan&scroll=3m&size={}".format(self.url,self.size)
            print(url)
            msg = urllib2.urlopen(url).read()
            self.writeFile(msg, out_file=out_file)
            start=start+1
        print("export data 完成!!!\n\t 总共耗时:"+str(time.time()-begin)+"s")

    def writeFile(self,msg,out_file=None):
        obj = json.loads(msg)
        vals = obj["hits"]["hits"]
        with open(out_file,"a",encoding='utf8')as f:
            for val in vals:
                a = json.dumps(val["_source"],ensure_ascii=False)
                print(a, file=f)



# http://172.26.1.128:9200/_plugin/kopf/#!/cluster
#查询接口：
# http://172.26.1.128:9200/_plugin/head/

# {"query":
#    {"match":
#      {"announcerName":"阿杜"}
#    }
# }
# 查看连接：
# gswyhq@gswyhq-pc:~$ curl 172.26.1.128:9200
#
# 查看集群的状态:
# gswyhq@gswyhq-pc:~$ curl -XGET 172.26.1.128:9200/_cluster/health?pretty
#
# # http://sg552.iteye.com/blog/1567047
#
# 查询所有的 index, type:
# gswyhq@gswyhq-pc:~$ curl 172.26.1.128:9200/_search?pretty=true
#
# 查询musicinfo这个index下的所有type:
# gswyhq@gswyhq-pc:~$ curl 172.26.1.128:9200/musicinfo/_search
#
# 查询ai_depart这个index下的ai_test这个type下的所有记录
# curl 172.26.1.128:9200/ai_depart/ai_test/_search?pretty=true
#
# 查询某个结果：
# curl 172.26.1.128:9200/musicinfo/music/_search?q=title:相容
#
# curl 172.26.1.128:9200/musicinfo/music/_search?q=url:http://resource.gow.cn/2016060114092322997773.mp3
#
# curl 172.26.1.128:9200/musicinfo/music/_search? -d ' {"query" : { "term": { "id":"207388"}}}'
#
# 查询ai_depart这个index下的ai_test这个type下的id为207388的记录：
# 在任意的查询字符串中增加pretty参数，类似于上面的例子。会让Elasticsearch美化输出(pretty-print)JSON响应以便更加容易阅读。_source字段不会被美化，它的样子与我们输入的一致。
# gswyhq@gswyhq-pc:~$ curl 172.26.1.128:9200/musicinfo/music/207388?pretty
#
# 请求个别字段：
# gswyhq@gswyhq-pc:~$ curl 172.26.1.128:9200/musicinfo/music/207388?_source=id,title
#
# 只想得到_source字段而不要其他的元数据
# gswyhq@gswyhq-pc:~$ curl 172.26.1.128:9200/musicinfo/music/207388?_source
#
#
# http://172.26.1.128:9200/musicinfo/music/_search?title=%22%E7%9B%B8%E5%AE%B9%22
#
#
# curl  -XGET '172.26.1.128:9200/musicinfo/music/_search?pretty' -d'
#     {
#         "query": {
#             "match": {
#                 "title": {
#                 "query":"山楂花",
#                 "operator":"and"
#                 }
#
#          }}
#    }
#    '
