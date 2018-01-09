#!/usr/bin/python3
# -*- coding:utf8 -*- #

import csv
from py2neo import Graph, Node, Relationship, Transaction, Subgraph

def expot_data(cid, data):
    """
    将数据导入到neo4j，给每个导入的实体添加一个标签cid.
    :param cid: 
    :param data: 
    :return: 
    """
    title = data[0]
    host, http_port, bolt_port, user, password = 'localhost', 7474, 7687, 'neo4j', 'gswyhq'
    graph = Graph(host=host, http_port=http_port, bolt_port=bolt_port, user=user, password=password)
    # title = ["_id", "_labels", "tagline", "title", "released", "name", "born", "_start", "_end", "_type", "roles"]
    _start_index = title.index('_start')
    node_property = title[2:_start_index]
    relation_property = title[_start_index + 3:]
    nodes = {}
    relationships = []
    tx = graph.begin()
    for line in data[1:]:
        _id, _labels = line[:2]
        node_property_value = line[2:_start_index]
        _start, _end, _type = line[_start_index:_start_index + 3]
        relation_property_value = line[_start_index + 3:]
        _labels = [label for label in _labels.strip().split(':') if label]
        _labels.append(cid.capitalize())
        # print(line)
        # nodes = {"a": Node("person", name="weiyudang", age=13), "b": Node("person", name="wangjiaqi")}
        if _id and not _start and not _end:
            property_dict = {k: v for k, v in zip(node_property, node_property_value) if v}
            _cid = "{}_{}".format(cid.lower(), _id)
            node = Node(*_labels, _cid=_cid, **property_dict)
            # graph.merge(node)
            nodes.setdefault(_cid, node)
            tx.create(node)
        elif not _id and _start and _end:
            property_dict = {k: v for k, v in zip(relation_property, relation_property_value) if v}
            start_cid = "{}_{}".format(cid.lower(), _start)
            end_cid = "{}_{}".format(cid.lower(), _end)
            # a = Node(_cid=start_cid)
            # b = Node(_cid=end_cid)
            a = nodes.get(start_cid)
            b = nodes.get(end_cid)
            a_knows_b = Relationship(a, _type, b, **property_dict)
            # graph.merge(a_knows_b)
            relationships.append(a_knows_b)
            tx.create(a_knows_b)
        else:
            raise ValueError("数据有误： {}".format(line))
    # sub_graph = Subgraph(nodes=nodes, relationships=relationships)
    # graph.create(sub_graph)
    tx.commit()

def read_data(file_path):

    with open(file_path, encoding='utf8') as csv_file:
        reader = csv.reader(csv_file)
        data = [row for row in reader]
    return data

def main():

    # 在web端（http://192.168.3.105:7474/browser/）导出neo4j数据到csv文件
    # 导出命令： CALL apoc.export.csv.all( "/var/lib/neo4j/data/all.csv", {})
    data = [
                # 标题行，依次是：节点id,节点类型，节点属性, ... ,关系的开始节点id, 关系的结束节点id, 关系的类型， 关系的属性, ...
                ['_id','_labels',  'tagline',  'title',  'released',  'name',  'born',  '_start',  '_end',  '_type',  'roles'],
                # 节点数据，若标题行对应的数据不存在，则为空：
                 ['0',  ':Movie',  'Welcome to the Real World',  'The Matrix',  '1999',  '',  '',  '',  '',  '',  ''],
                 ['1', ':Person', '', '', '', 'Keanu Reeves', '1964', '', '', '', ''],
                 ['18', ':Person', '', '', '', 'Carrie-Anne Moss', '1967', '', '', '', ''],
                 ['19', ':Person', '', '', '', 'Laurence Fishburne', '1961', '', '', '', ''],
                 ['20', ':Person', '', '', '', 'Hugo Weaving', '1960', '', '', '', ''],
                 ['21', ':Person', '', '', '', 'Lilly Wachowski', '1967', '', '', '', ''],
                 ['22', ':Person', '', '', '', 'Lana Wachowski', '1965', '', '', '', ''],
                 ['23', ':Person', '', '', '', 'Joel Silver', '1952', '', '', '', ''],
                # 关系数据，若标题行对应的数据不存在，则为空：
                 ['', '', '', '', '', '', '', '1', '0', 'ACTED_IN', '["Neo"]'],
                 ['', '', '', '', '', '', '', '18', '0', 'ACTED_IN', '["Trinity"]'],
                 ['', '', '', '', '', '', '', '19', '20', 'ACTED_IN', '["Morpheus"]'],
                 ['', '', '', '', '', '', '', '20', '0', 'ACTED_IN', '["Agent Smith"]'],
                 ['', '', '', '', '', '', '', '21', '0', 'DIRECTED', ''],
                 ['', '', '', '', '', '', '', '22', '0', 'DIRECTED', ''],
                 ['', '', '', '', '', '', '', '23', '0', 'PRODUCED', '']
                 ]

    # data = read_data(file_path)
    expot_data('abc', data)

if __name__ == '__main__':
    main()