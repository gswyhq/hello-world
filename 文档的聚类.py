#----coding:utf-8----------------------------------------------------------------
# 名称: 文档的聚类
# 目的:
# http://stackoverflow.com/questions/1789254/clustering-text-in-python
# 作者:      gswyhq
#
# 日期:      2016-01-05
# 版本:      Python 3.3.5
# 系统:      win32
# Email:     gswyhq@126.com
#-------------------------------------------------------------------------------


from math import log, sqrt
from itertools import combinations
import jieba

def cosine_distance(a, b):
    """计算两篇文章特征向量间的余弦距离，返回值越大越相似。"""
    cos = 0.0
    a_tfidf = a["tfidf"]
    for token, tfidf in b["tfidf"].items():
        if token in a_tfidf:
            cos += tfidf * a_tfidf[token]
    return cos

def normalize(features):
    """将数据进行单位化。一个非零向量除以它的模,可得所需单位向量。"""
    norm = 1.0 / sqrt(sum(i**2 for i in features.values()))#python3中不存在dict.itervalues()用法
    for k, v in features.items():
        features[k] = v * norm
    return features

def add_tfidf_to(documents):
    """计算每篇文章特征的TF-IDF加权"""
    tokens = {}#统计所有的文章
    for id, doc in enumerate(documents):
        tf = {}#统计每个doc中的单词数，TF词频(Term Frequency)，TF表示词条在文档d中出现的频率。
        doc["tfidf"] = {}
        doc_tokens = doc.get("tokens", [])#dict.get(key, default=None)参数key -- 这是要搜索在字典中的键。default -- 这是要返回键不存在的的情况下默认值。
        for token in doc_tokens:
            tf[token] = tf.get(token, 0) + 1
        num_tokens = len(doc_tokens)
        if num_tokens > 0:
            for token, freq in tf.items():
                tokens.setdefault(token, []).append((id, float(freq) / num_tokens))#词频
        #setdefault()方法类似于get()方法，但会设置字典[键]=默认情况下，如果键不是已经在字典中。dict.setdefault(key, default=None);
        #参数    key -- 这是要被搜索的键    default -- 这是没有找到键的情况下返回的值。此方法返回字典可用的键值，如果给定键不可用，则它会返回所提供的默认值。

    doc_count = float(len(documents))#总文章数
    for token, docs in tokens.items():
        idf = log(doc_count / len(docs))#单词出现文章比例的负对数，即出现的文章越多其对应的值越小
        #IDF逆向文件频率(Inverse Document Frequency)，
        #由总文件数目除以包含该词语之文件的数目，再将得到的商取对数，其等价于：由包含该词语之文件的数目除以总文件数目，再将得到的商取负对数
        for id, tf in docs:
            tfidf = tf * idf
            #如果某个词或短语在一篇文章中出现的频率TF高，并且在其他文章中很少出现，则认为此词或者短语具有很好的类别区分能力，适合用来分类
            #某一特定文件内的高词语频率，以及该词语在整个文件集合中的低文件频率，可以产生出高权重的TF-IDF
            if tfidf > 0:
                documents[id]["tfidf"][token] = tfidf

    for doc in documents:
        doc["tfidf"] = normalize(doc["tfidf"])#单位化处理

def choose_cluster(node, cluster_lookup, edges):
    '''在edges中查找与node最为相似的文章，并返回文章的id.'''
    new = cluster_lookup[node]
    if node in edges:
        seen, num_seen = {}, {}
        #获取与node相似的{文章：距离}字典
        for target, weight in edges.get(node, []):
            seen[cluster_lookup[target]] = seen.get(
                cluster_lookup[target], 0.0) + weight
        for k, v in seen.items():#将距离相近的添加到num_seen（以距离值作为键的字典）
            num_seen.setdefault(v, []).append(k)
        new = num_seen[max(num_seen)][0]#返回第一个最为相似的
    return new

def majorclust(graph):
    '''根据graph.edges提供的距离，对文档graph.nodes进行聚类，返回聚类后的文档,如：
    dict_values([[0, 1, 3], [2, 4], [5, 6, 7]])
    '''
    #cluster_lookup:用于文章所属的聚类
    #初始化，最近距离的文章是自己
    cluster_lookup = dict((node, i) for i, node in enumerate(graph.nodes))
    #cluster_lookup：{0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7}
    count = 0
    movements = set()
    finished = False
    while not finished:
        finished = True
        for node in graph.nodes:
            new = choose_cluster(node, cluster_lookup, graph.edges)#查找与node距离最近的文章
            move = (node, cluster_lookup[node], new)
            if new != cluster_lookup[node] and move not in movements:
                movements.add(move)
                cluster_lookup[node] = new #将与node最近的文章更新
                finished = False #若更新的一个，则需要全面检查一遍是否可以继续更新

    clusters = {}
    for k, v in cluster_lookup.items():
        clusters.setdefault(v, []).append(k)

    return clusters.values()

def get_distance_graph(documents):
    """计算文章间的距离"""
    class Graph(object):
        def __init__(self):
            self.edges = {}

        def add_edge(self, n1, n2, w):
            self.edges.setdefault(n1, []).append((n2, w))
            self.edges.setdefault(n2, []).append((n1, w))

    graph = Graph()
    doc_ids = range(len(documents))
    graph.nodes = set(doc_ids)# 数据属性不需要预先定义!当数据属性初次被使用时,它即被创建并赋值.而实际上,类属性也是如此.
    #python的class中有两种属性:类属性,数据属性.类属性属于类,数据属性属于类的实例。"改变类属性,数据属性跟着变,改变数据属性,类属性不变"
    for a, b in combinations(doc_ids, 2):
        # combinations('ABCD', 2) --> AB AC AD BC BD CD
        # combinations(range(4), 3) --> 012 013 023 123
        graph.add_edge(a, b, cosine_distance(documents[a], documents[b]))
    return graph

def split_word(text):
    '''利用jieba对text进行分词，返回所得分词组成的list.'''
    split_text=jieba.cut(text, cut_all=0)#不全分类

    #读取样本文本
    #去除停用词，同时构造样本词的字典
    with open(r'f:\python\data\stopwords.txt',encoding='utf-8') as f:
        stop_text = f.read( )
    f_stop_seg_list=stop_text.split('\n')
    new_text=[t for t in split_text if (t not in f_stop_seg_list)and len(t)>1]
    return ' '.join(new_text)

def 解析中文():
    texts=['深圳龙华超市','深圳初中关外','北京雾霾天安门','北京长城故宫','初中教师语文']
    texts=['另外，龙华现代有轨电车试验线总投资约24 .2亿元，已完成投资1 .95亿元，主体工程完成2 .3%，同步实施的道路改造、绿化迁移、管线改迁等工程完成19.3%，力争于2017年1月开通试运行。深圳同步开展运营管理立法，正在组织研究制订《有轨电车驾驶资格及车辆牌证管理办法》、《有轨电车沿线道路通行秩序管理办法》、《现代有轨电车运营服务管理办法》。相关管理办法已完成初稿，力争2016年底前完成立法工作。',
    '根据深圳轨道交通三期工程建设规划，预计到2020年，深圳轨道交通将形成11条完整线路、435公里轨道交通网络。深圳就此形成16个项目，总长411公里，拟总投资3248亿元的建议上报方案。目前正进一步梳理各专家、各区政府的意见，修改完善方案，力争今年一季度形成轨道交通四期建设规划方案上报国家有关部门审批。',
    '2015年市政府加大财政直接投资支持力度，拨付679 .96亿元财政资金给地铁集团，是历年来支持力度最大的一年。同时，市轨道办会同市交通运输委、财政委等有关部门，正在综合考虑深圳运营和财政实际情况，研究完善深圳轨道交通运营补贴机制。按照《办理方案》的要求，市政府还将于2016年底向市人大常委会报告该项重点建议的办理情况。',
    '近日，网友拍摄到了在船厂里的中国第2艘万吨级海警执法船，该船已经完成建造，整装待发，即将加入到中国海上执维权执法的序列中，未来或部署南海。该船编号3901的海警船满载排水量超过万吨，将成为我国海上执法力量的中流砥柱。另据报道，首艘同型万吨海警船2901号在建造完成后，已经进驻舟山。',
    '长期以来，中国的海上执法力量一直处于劣势地位。海上执法船吨位太小，火力太弱，再加上“九龙闹海”的局面，所以常常遭受一些国家的欺凌。曾几何时，为了维护自己的海洋权益，中国不得不派遣军舰同对方的执法船对峙。这种做法一方面容易授人以口实，另一方面不好把控冲突的烈度。近年来，中国海上执法力量励精图治，不但结束了力量分散的局面，而且建造了一大批优秀的海上执法船，其中就包括本文的主角海警2901船。',
    '近日，海警2901船已经开始海试。该船有两大特点，一是吨位大，二是火力强。海警2901船在首部安装了76毫米舰炮，同美国现役“汉密尔顿”级巡逻舰安装的火炮属于同一口径。此外，该船的吨位达到了惊人的1.2万吨，取代“敷岛”级成为世界上最大的海警船。'
    ]

    return [{"text": i, "tokens": split_word(text)}
             for i, text in enumerate(texts)]

def get_documents():
    texts = [
        "foo blub baz",
        "foo bar baz",
        "asdf bsdf csdf",
        "foo bab blub",
        "csdf hddf kjtz",
        "123 456 890",
        "321 890 456 foo",
        "123 890 uiop",
    ]

    return [{"text": (i,text), "tokens": text.split()}
             for i, text in enumerate(texts)]

def main():
    documents = get_documents()#读取文本，返回格式为：包括"text"键和"tokens"键的字典的列表，如下：
    '''
[{'text': 'foo blub baz', 'tokens': ['foo', 'blub', 'baz']},
 ...,
 {'text': '123 890 uiop', 'tokens': ['123', '890', 'uiop']}]
    '''
    documents=解析中文()
    add_tfidf_to(documents)# 计算 TF-IDF加权，即在字典参数documents中添加一个'tfidf'键,值是文章单词及其权重组成的字典,如下：
    '''
 [{'text': 'foo blub baz',
  'tfidf': {'baz': 0.6666666666666666,
            'blub': 0.6666666666666666,
            'foo': 0.3333333333333333},
  'tokens': ['foo', 'blub', 'baz']},
 ...,
 {'text': '123 890 uiop',
  'tfidf': {'123': 0.5163576579224811,
            '890': 0.36533272450005416,
            'uiop': 0.7745364868837216},
  'tokens': ['123', '890', 'uiop']}]
    '''
    dist_graph = get_distance_graph(documents)# 计算文章间的距离,返回一个自定义graph类的实例，其属性包括：edges和nodes
    '''
    dist_graph.edges的结构如下：
{0: [(1, 0.44543540318737407),
     (2, 0.0),
     (3, 0.44543540318737407),
     (4, 0.0),
     (5, 0.0),
     (6, 0.08332726336023205),
     (7, 0.0)],
...,
 7: [(0, 0.0),
     (1, 0.0),
     (2, 0.0),
     (3, 0.0),
     (4, 0.0),
     (5, 0.4899930981962787),
     (6, 0.12923046242945363)]}

     dist_graph.nodes的内容如下：
{0, 1, 2, 3, 4, 5, 6, 7}
    '''

    for cluster in majorclust(dist_graph):
        print ("========="*10)
        for doc_id in cluster:
            print (documents[doc_id]["text"])

if __name__ == '__main__':
    main()