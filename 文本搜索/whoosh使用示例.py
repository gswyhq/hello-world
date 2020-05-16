#!/usr/bin/python3
# coding: utf-8

import os
import json
from whoosh.index import create_in, open_dir
from whoosh.fields import Schema, ID, TEXT
from whoosh.qparser import QueryParser, MultifieldParser, syntax
from whoosh.query import FuzzyTerm
from jieba.analyse import ChineseAnalyzer

WHOOSH_INDEXER_PATH = '/home/gswyhq/Downloads/whoosh_/indexer'

schema = Schema(title = TEXT(stored = True, analyzer=ChineseAnalyzer()),path = ID(stored=True),content=TEXT(stored=True, analyzer=ChineseAnalyzer()))

if not os.path.isdir(WHOOSH_INDEXER_PATH):
    print('若路径不存在，则创建路径')
    os.makedirs(WHOOSH_INDEXER_PATH)
if not os.listdir(WHOOSH_INDEXER_PATH):
    print('若路径为空，则创建索引')
    ix = create_in(WHOOSH_INDEXER_PATH, schema)
else:
    print('加载已有索引：{}'.format(WHOOSH_INDEXER_PATH))
    ix = open_dir(WHOOSH_INDEXER_PATH, schema=schema)

data = '''首春:寒随穷律变，春逐鸟声开。初风飘带柳，晚雪间花梅。碧林青旧竹，绿沼翠新苔。芝田初雁去，绮树巧莺来。
初晴落景:晚霞聊自怡，初晴弥可喜。日晃百花色，风动千林翠。池鱼跃不同，园鸟声还异。寄言博通者，知予物外志。
初夏:一朝春夏改，隔夜鸟花迁。阴阳深浅叶，晓夕重轻烟。哢莺犹响殿，横丝正网天。珮高兰影接，绶细草纹连。碧鳞惊棹侧，玄燕舞檐前。何必汾阳处，始复有山泉。
度秋:夏律昨留灰，秋箭今移晷。峨嵋岫初出，洞庭波渐起。桂白发幽岩，菊黄开灞涘。运流方可叹，含毫属微理。
仪鸾殿早秋:寒惊蓟门叶，秋发小山枝。松阴背日转，竹影避风移。提壶菊花岸，高兴芙蓉池。欲知凉气早，巢空燕不窥。
秋日即目:爽气浮丹阙，秋光澹紫宫。衣碎荷疏影，花明菊点丛。袍轻低草露，盖侧舞松风。散岫飘云叶，迷路飞烟鸿。砌冷兰凋佩，闺寒树陨桐。别鹤栖琴里，离猿啼峡中。落野飞星箭，弦虚半月弓。芳菲夕雾起，暮色满房栊。
山阁晚秋:山亭秋色满，岩牖凉风度。疏兰尚染烟，残菊犹承露。古石衣新苔，新巢封古树。历览情无极，咫尺轮光暮。'''

def add_document(ix, data):
    writer = ix.writer()
    data = [t.split(':', maxsplit=1) for t in data.split('\n')]
    print('新增数据')
    for title, content in data:
        writer.add_document(title=title, content = content)
    writer.commit()

# add_document(ix, data)

class MyFuzzyTerm(FuzzyTerm):
    def __init__(self, fieldname, text, boost=1.0, maxdist=5,
                 prefixlength=1, constantscore=True):
        super(MyFuzzyTerm, self).__init__(fieldname, text, boost, maxdist, prefixlength, constantscore)

with ix.searcher() as searcher:
    # group=syntax.OrGroup， 匹配多个关键词中的一个
    query = QueryParser("content", ix.schema, termclass=FuzzyTerm, group=syntax.OrGroup).parse("山泉古树晚霞")
    # 分析query,生成query对象
    print("query:{}".format(query))
    results = searcher.search(query, limit=10) # 若要得到全部的结果，可把limit=None.

    print('一共发现%d份文档。' % len(results))
    for i in range(min(10, len(results))):
        print(json.dumps(results[i].fields(), ensure_ascii=False))

    results = searcher.find("content", "古树山泉")
    print('一共发现%d份文档。' % len(results))

def main():
    pass


if __name__ == '__main__':
    main()