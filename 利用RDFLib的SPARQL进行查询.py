#!/usr/bin/python
# -*- coding:utf8 -*- #
from __future__ import generators
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import sys

if sys.version >= '3':
    PY3 = True
else:
    PY3 = False

if PY3:
    import pickle
else:
    import cPickle as pickle
    from codecs import open

import rdflib

def main():


    g = rdflib.Graph()
    has_border_with = rdflib.URIRef('http://www.example.org/has_border_with')
    located_in = rdflib.URIRef('http://www.example.org/located_in')

    germany = rdflib.URIRef('http://www.example.org/country1')
    france = rdflib.URIRef('http://www.example.org/country2')
    china = rdflib.URIRef('http://www.example.org/country3')
    mongolia = rdflib.URIRef('http://www.example.org/country4')

    europa = rdflib.URIRef('http://www.example.org/part1')
    asia = rdflib.URIRef('http://www.example.org/part2')

    g.add((germany,has_border_with,france))
    g.add((china,has_border_with,mongolia))
    g.add((germany,located_in,europa))
    g.add((france,located_in,europa))
    g.add((china,located_in,asia))
    g.add((mongolia,located_in,asia))

    for s,p,o in g:
        print(s,p,o,sep='       ')
    print ('--------------------------------------------------')

    #q = "select ?country where { ?country <http://www.example.org/located_in> <http://www.example.org/part1> }"
    w = """select ?xyz
        where {
        ?xyz
        <http://www.example.org/has_border_with>
        <http://www.example.org/country2>
        }"""
    print(list(g.query(w)),'&'*20)


    q = """select ?xyz
        where {
        <http://www.example.org/country3>
        <http://www.example.org/has_border_with>
        ?xyz
        }"""
    x = g.query(q)
    print (list(x))

    # 写图表文件，重新阅读和查询新创建的图
    #序列化图表到文件,内置格式有：'xml', 'n3', 'turtle', 'nt', 'pretty-xml', 'trix', 'trig' and 'nquads'
    #g.serialize("graph.rdf",format('nt'))
    g1 = rdflib.Graph()
    #g1.parse("/home/gswewf/appliances/graph.rdf", format="nt")#根据自己的上下文解析新增三元组
    g1.load('graph.rdf',format='nt') #load：加载完，送给parse操作，同parse
    x1 = g1.query(q)
    print ('*'*8,list(x1))

    #g = rdflib.Graph()

    #g.parse("vc-db-1.rdf")
    print ('--------------------------------------------------')
    q = "SELECT ?x WHERE { ?x  <http://www.example.org/has_border_with> <http://www.example.org/country4> }"
    qres = g.query(q)
    print (qres.vars)
    # [rdflib.term.Variable(u'givenName')]
    print (qres.bindings)
    #qres.bindings中含有我们需要的结果，如果值为空，则输出[]，所以我们只需要令qres.bindings==[]即可判断。
    # [{rdflib.term.Variable(u'givenName'): rdflib.term.Literal(u'Rebecca')}, {rdflib.term.Variable(u'givenName'): rdflib.term.Literal(u'John')}]
    print (qres.graph )

if __name__ == "__main__":
    main()

'''
from rdflib.query import Result

class SPARQLResult(Result):

    def __init__(self, res):
        Result.__init__(self, res["type_"])
        self.vars = res.get("vars_")
        self.bindings = res.get("bindings")
        self.askAnswer = res.get("askAnswer")
        self.graph = res.get("graph")



from rdflib.collection import Collection
g=Graph()
a=BNode('foo')
b=BNode('bar')
c=BNode('baz')
g.add((a,RDF.first,RDF.type))
g.add((a,RDF.rest,b))
g.add((b,RDF.first,RDFS.label))
g.add((b,RDF.rest,c))
g.add((c,RDF.first,RDFS.comment))
g.add((c,RDF.rest,RDF.nil))
def topList(node,g):
   for s in g.subjects(RDF.rest,node):
      yield s
def reverseList(node,g):
   for f in g.objects(node,RDF.first):
      print(f)
   for s in g.subjects(RDF.rest,node):
      yield s


        '''
