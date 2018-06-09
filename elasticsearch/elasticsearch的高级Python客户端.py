#!/usr/bin/python3
# coding: utf-8

# 兼容性
# 该库与自1.x以来的所有Elasticsearch版本兼容，但您必须使用匹配的主要版本：
# 对于Elasticsearch 6.0和更高版本，使用库的主版本5（6.x.y）。
# 对于Elasticsearch 5.0和更高版本，使用库的主版本5（5.x.y）。
# 对于Elasticsearch 2.0和更高版本，使用库的主要版本2（2.x.y）。
# 建议您在setup.py或requirements.txt中设置要求的方法是：
#
# ＃Elasticsearch 6.x
# elasticsearch-DSL>=6.0.0，<7.0.0
#
# ＃Elasticsearch 5.x
# elasticsearch-DSL>=5.0.0，<6.0.0
#
# ＃Elasticsearch 2.x
# elasticsearch-DSL>=2.0.0，<3.0.0

# gswewf@gswewf-PC:~$ curl http://192.168.3.105:9200/
# {
#   "name" : "CGpcm3o",
#   "cluster_name" : "elasticsearch",
#   "cluster_uuid" : "_2rxVeOMSbK0_eyc514aUQ",
#   "version" : {
#     "number" : "5.6.4",
#     "build_hash" : "8bbedf5",
#     "build_date" : "2017-10-31T18:55:38.105Z",
#     "build_snapshot" : false,
#     "lucene_version" : "6.6.1"
#   },
#   "tagline" : "You Know, for Search"
# }
# Elasticsearch版本是5.6.4，故：
# gswewf@gswewf-PC:~$ sudo pip3 install elasticsearch-dsl==5.4.0

# 正常的一个查询示例：
def abc_test1():
    from elasticsearch import Elasticsearch
    client = Elasticsearch()

    response = client.search(
        index="my-index",
        body={
          "query": {
            "bool": {
              "must": [{"match": {"title": "python"}}],
              "must_not": [{"match": {"description": "beta"}}],
              "filter": [{"term": {"category": "search"}}]
            }
          },
          "aggs" : {
            "per_tag": {
              "terms": {"field": "tags"},
              "aggs": {
                "max_lines": {"max": {"field": "lines"}}
              }
            }
          }
        }
    )

    for hit in response['hits']['hits']:
        print(hit['_score'], hit['_source']['title'])

    for tag in response['aggregations']['per_tag']['buckets']:
        print(tag['key'], tag['max_lines']['value'])

# 用elasticsearch-dsl，改写一下上示例
def abc_test2():
    from elasticsearch import Elasticsearch
    from elasticsearch_dsl import Search

    client = Elasticsearch()

    s = Search(using=client, index="my-index") \
        .filter("term", category="search") \
        .query("match", title="python")   \
        .exclude("match", description="beta")

    s.aggs.bucket('per_tag', 'terms', field='tags') \
        .metric('max_lines', 'max', field='lines')

    response = s.execute()

    for hit in response:
        print(hit.meta.score, hit.title)

    for tag in response.aggregations.per_tag.buckets:
        print(tag.key, tag.max_lines.value)

# 如你所见：
#     按名称创建适当的查询对象（例如“match”）
#     将查询组合成复合布尔查询
#     将术语查询放在bool查询的过滤器上下文中
#     提供方便的访问响应数据
#     到处都没有卷曲或方括号

# 更多示例：

from datetime import datetime
from elasticsearch_dsl import DocType, Date, Integer, Keyword, Text
from elasticsearch_dsl.connections import connections

# 定义一个默认的 Elasticsearch 客户端
connections.create_connection(hosts=['localhost'])

# 类名“Article”的小写, 即为类型名 _type：“article”
class Article(DocType):
    # 定义各个字段的索引
    title = Text(analyzer='snowball', fields={'raw': Keyword()})
    body = Text(analyzer='snowball')
    tags = Keyword()
    published_from = Date()
    lines = Integer()

    class Meta:
        # 定义索引名, 类型名
        index = 'blog'
        doc_type = 'gswewf'  # 若无类型名设置，则采用类名“Article”的小写为类型名

    def save(self, ** kwargs):
        # kwargs['using'] = 'gswewf_alias'
        self.lines = len(self.body.split())  # 在当前记录上继续添加字段名：lines, 字段值为2
        return super(Article, self).save(** kwargs)

    def is_published(self):
        return datetime.now() > self.published_from

# 创建 the mappings in elasticsearch
Article.init()
# {'article': {'properties': {'body': {'analyzer': 'snowball', 'type': 'text'},
#    'lines': {'type': 'integer'},
#    'published_from': {'type': 'date'},
#    'tags': {'type': 'keyword'},
#    'title': {'analyzer': 'snowball',
#     'fields': {'raw': {'type': 'keyword'}},
#     'type': 'text'}}}}

# 创建并保存文章
# 创建一条id为42，字段名为：title，值为'Hello world!'的记录；
article = Article(meta={'id': 42}, title='Hello world!', tags=['test'])
article.body = ''' looong text '''  # 在刚才这条记录上继续添加字段名：body, 字段值为“ looong text ”
article.published_from = datetime.now()  #  在刚才这条记录上继续添加字段名：published_from, 字段值为“2018-01-27T10:35:19.455549”
article.save()

article = Article.get(id=42)
print(article.is_published())

# 查看集群的运行状况
print(connections.get_connection().cluster.health())


# 在这个例子中：
#     提供默认连接
#     用映射配置定义字段
#     设置索引名称
#     定义自定义方法
#     重写内置的.save（）方法以挂钩到持久生命周期
#     将对象检索并保存到Elasticsearch中

# 你不需要移植你的整个应用程序来获得Python DSL的好处，你可以从现有的dict中创建一个Search对象开始，使用API​​修改它，然后将它序列化为一个字典：

# 复杂的查询
body = {
        "query": {
            "match": {
                "query_question": {
                    "query": "恢复的效力你知道是什么意思啊",
                    "minimum_should_match": "30%"
                }
            }
        },
        "size": 20
    }

# 转换为搜索对象
s = Search.from_dict(body)

# 添加一些过滤器，聚合，查询，...
s.filter("term", tags="python")

# 转换回字典插回到现有的代码
body = s.to_dict()

def main():
    pass


if __name__ == '__main__':
    main()
