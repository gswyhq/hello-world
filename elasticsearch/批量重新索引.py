#!/usr/bin/python3
# coding: utf-8

import requests
from elasticsearch import Elasticsearch
from elasticsearch import helpers
from datetime import datetime

# 保险百科, 保险名词解释
BAOXIAN_INDEX_MAPPINGS_AND_SETTINGS = {
    "settings": {
        "analysis": {
            "filter": {
                "my_synonym_filter": {
                    "type": "synonym",  # 首先，我们定义了一个 同义词 类型的语汇单元过滤器。
# 同义词可以使用 synonym 参数来内嵌指定，或者必须 存在于集群每一个节点上的同义词文件中。 同义词文件路径由 synonyms_path 参数指定，应绝对或相对于 Elasticsearch config 目录。
# 同义词格式, 同义词最简单的表达形式是 逗号分隔; 或者, 使用 => 语法，可以指定一个词项列表（在左边），和一个或多个替换（右边）的列表：
# https://www.elastic.co/guide/cn/elasticsearch/guide/cn/synonym-formats.html
                    "synonyms": [
                        "未到期责任准备金,未赚保费准备金",
                        "人寿保险责任准备金,人身保险责任准备金",
                        "自愿保险,任意保险",
                        "履约保证保险,履约保证险,履约责任保险"
                    ]
                }
            },
            "analyzer": {
                "my_synonyms": {
                    "tokenizer": "ik_max_word",
                    "filter": [
                        "lowercase",
                        "my_synonym_filter"  # 创建了一个使用 my_synonym_filter 的自定义分析器。
                    ]
                }
            }
        }
    },
    "mappings": {
        'baoxian':
            {'properties': {
                'question': {'analyzer': 'my_synonyms',
                             'index': 'analyzed',
                             'type': 'string'},
                'answer': {'analyzer': 'ik_max_word',
                           'index': 'analyzed',  # index: "no"   #不分词，不索引; "analyze" #分词,索引; "not_analyzed" # 不去分词
                           'type': 'string'},
                'description': {'type': 'string'},
                'timestamp': {
                    # 'format': "yyyy-MM-dd HH:mm:ss",
                    'type': 'date'}
                }
                }
            },
}

ES_HOST = '192.168.3.250'
ES_PORT = '9200'

ES_SNIFFER_TIMEOUT = 120  # 检查节点状态的间隔时间，单位：秒；启动或故障时，检查集群状态以获取节点列表
ES_MAXSIZE = 25  # 允许连接到每个节点的线程数
ES_REQUEST_TIMEOUT = 3  # 每个请求的超时设置，单位：秒


old_index = 'baike'
_index = 'baike_v3'
_doc_type = 'baoxian'
alias_name = 'baike_alias'

class ReIndex():
    
    def __init__(self):
        self.es = Elasticsearch(hosts=[{"host": ES_HOST, "port": ES_PORT}],
                sniff_on_start=True,  # 启动时检查集群状态以获取节点列表
                sniff_on_connection_fail=True,  # 故障时，检查集群状态以获取节点列表
                sniffer_timeout=ES_SNIFFER_TIMEOUT,  # 每隔段时间检查一下节点状态； 节点无效响应后刷新节点
                maxsize=ES_MAXSIZE,  # 允许连接到每个节点的线程数
                request_timeout=ES_REQUEST_TIMEOUT,
                                    )

    def create_index(self, _index, alias_name, body=BAOXIAN_INDEX_MAPPINGS_AND_SETTINGS):
        self.es.indices.create(index=_index, ignore=400, body=body)  # 创建索引
        if self.es.indices.exists_alias(name=alias_name):
            # 判断某个别名是否已经存在
            # alias_dict = self.es.indices.get_alias(name=alias_name)  # 检测这个别名指向哪一个索引
            # alias_dict = self.es.indices.get_alias(index=_index)  # 检测哪些别名指向这个索引
            # alias_dict = self.es.indices.get_alias(name)  # 根据索引，或别名，查询对应的指向关系
            # 返回数据结构都是： {'baike_v2': {'aliases': {'baike_alias': {}}},'baike_v3': {'aliases': {'baike_alias': {}}}}
            # index_list = [k for k in alias_dict.keys()]
            # self.es.indices.delete_alias(index=index_list, name=alias_name)  # 删除索引及别名的指向
            self.es.indices.delete_alias(index='_all', name=alias_name)  # 删除别名指向所有的索引

        self.es.indices.put_alias(_index, name=alias_name)  # 给索引取别名

    def bulk_add_data(self, _index, _doc_type, dict_data=None):
        """
        批量添加数据
        :param _index: 索引名
        :param _doc_type: 类型名
        :param dict_data: 数据列表
        :return:
        """

        if not dict_data:
            dict_data = []
        # j = 0
        actions = []
        for value in dict_data:
            action = {
                "_index": _index,
                "_type": _doc_type,
                # "_id": j + 1,
                "_source": {
                    "timestamp": datetime.now()}
            }
            action["_source"].update(value)
            actions.append(action)
            # j += 1


            if (len(actions) == 500000):
                helpers.bulk(self.es, actions)
                del actions[0:len(actions)]

        if (len(actions) > 0):
            helpers.bulk(self.es, actions)

def initialization(es_host='192.168.3.250', es_port='9200', old_index='', _doc_type=''):
    """
    初始化
    初始化时需要像普通 search 一样，指明 index 和 type (当然，search 是可以不指明 index 和 type 的)，然后，加上参数 scroll，表示暂存搜索结果的时间，其它就像一个普通的search请求一样。
    初始化返回一个 _scroll_id，_scroll_id 用来下次取数据用。
    :return: 
    """
    url = 'http://{}:{}/{}/{}/_search?scroll=10m'.format(es_host, es_port, old_index, _doc_type)
    body = {"query": { "match_all": {}
                       },
            "sort": ["_doc"],
            "size":  2000,
            }
    ret = requests.post(url, json=body)
    json_ret = ret.json()
    _scroll_id = json_ret.get('_scroll_id')
    hits_source = [hits.get('_source') for hits in json_ret.get('hits', {}).get('hits', []) if hits.get('_source')]
    return _scroll_id, hits_source


def traversal(es_host='192.168.3.250', es_port='9200', scroll_id=''):
    """
    这里的 scroll_id 即 上一次遍历取回的 _scroll_id 或者是初始化返回的 _scroll_id，同样的，需要带 scroll 参数。 
    重复这一步骤，直到返回的数据为空，即遍历完成。注意，每次都要传参数 scroll，刷新搜索结果的缓存时间。另外，不需要指定 index 和 type。
    :param es_host: 
    :param es_port: 
    :return: 
    """
    url = 'http://{}:{}/_search/scroll?scroll=10m'.format(es_host, es_port)
    body = {
        "scroll_id": scroll_id,
            }
    ret = requests.post(url, json=body)
    json_ret = ret.json()
    _scroll_id = json_ret.get('_scroll_id')
    hits_source = [hits.get('_source') for hits in json_ret.get('hits', {}).get('hits', []) if hits.get('_source')]
    return _scroll_id, hits_source

def update_index_aliase(es_host, es_port, old_index, new_index, index_aliase):
    """
    一个别名可以指向多个索引，所以我们在添加别名到新索引的同时必须从旧的索引中删除它。这个操作需要原子化，这意味着我们需要使用
    _aliases
    :param es_host: 
    :param es_port: 
    :param old_index: 
    :param new_index: 
    :param index_aliase: 
    :return: 
    """
    url = 'http://{}:{}/_aliases'.format(es_host, es_port)
    body = {
        "actions": [
            {"remove": {"index": old_index, "alias": index_aliase}},
            {"add": {"index": new_index, "alias": index_aliase}}
        ]
    }
    ret = requests.post(url, json=body)
    json_ret = ret.json()
    print(json_ret)

def main():
    re_index = ReIndex()
    re_index.create_index(_index, alias_name, body=BAOXIAN_INDEX_MAPPINGS_AND_SETTINGS)

    _scroll_id, hits_source = initialization(es_host=ES_HOST, es_port=ES_PORT, old_index=old_index, _doc_type=_doc_type)
    # print(_scroll_id, hits_source)
    re_index.bulk_add_data(_index, _doc_type, dict_data=hits_source)
    while(hits_source):
        _scroll_id, hits_source = traversal(es_host=ES_HOST, es_port=ES_PORT, scroll_id=_scroll_id)
        # print(_scroll_id, hits_source)
        print(len(hits_source))
        re_index.bulk_add_data(_index, _doc_type, dict_data=hits_source)

    update_index_aliase(es_host=ES_HOST, es_port=ES_PORT, old_index=old_index, new_index=_index, index_aliase=alias_name)

if __name__ == '__main__':
    main()