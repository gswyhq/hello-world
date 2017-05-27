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
from rdflib.graph import Graph,QuotedGraph,ConjunctiveGraph,Dataset,Store,ReadOnlyGraphAggregate,Namespace
from rdflib import RDF,URIRef,BNode,RDFS,Literal,plugin, Variable
from rdflib.query import ResultRow
g = Graph()
statementId = BNode()
print(len(g))
g.add((statementId, RDF.type, RDF.Statement))
g.add((statementId, RDF.subject,
     URIRef('http://rdflib.net/store/ConjunctiveGraph')))
g.add((statementId, RDF.predicate, RDFS.label))
g.add((statementId, RDF.object, Literal("Conjunctive Graph")))
print(len(g))

for s, p, o in g:
    print(s,'----',p,'----',o)

#N400459b56eef41519110dade575e34f0 ---- http://www.w3.org/1999/02/22-rdf-syntax-ns#predicate ---- http://www.w3.org/2000/01/rdf-schema#label
#N400459b56eef41519110dade575e34f0 ---- http://www.w3.org/1999/02/22-rdf-syntax-ns#subject ---- http://rdflib.net/store/ConjunctiveGraph
#N400459b56eef41519110dade575e34f0 ---- http://www.w3.org/1999/02/22-rdf-syntax-ns#object ---- Conjunctive Graph
#N400459b56eef41519110dade575e34f0 ---- http://www.w3.org/1999/02/22-rdf-syntax-ns#type ---- http://www.w3.org/1999/02/22-rdf-syntax-ns#Statement
for s, p, o in g.triples((None, RDF.object, None)):
    print(o)



g1 = Graph()
g2 = Graph()
u = URIRef('http://example.com/foo')
g1.add([u, RDFS.label, Literal('foo')])
g1.add([u, RDFS.label, Literal('bar')])
g2.add([u, RDFS.label, Literal('foo')])
g2.add([u, RDFS.label, Literal('bing')])
print(len(g1 + g2))  # adds bing as label,3

print(len(g1 - g2))  # 删除 foo,1

print(len(g1 * g2))  # 交集 foo,1

g1 += g2  # 合并
print(len(g1))


store = plugin.get('IOMemory', Store)()
g1 = Graph(store)
g2 = Graph(store)
g3 = Graph(store)
stmt1 = BNode()
stmt2 = BNode()
stmt3 = BNode()
g1.add((stmt1, RDF.type, RDF.Statement))
g1.add((stmt1, RDF.subject,
     URIRef('http://rdflib.net/store/ConjunctiveGraph')))
g1.add((stmt1, RDF.predicate, RDFS.label))
g1.add((stmt1, RDF.object, Literal("Conjunctive Graph")))
g2.add((stmt2, RDF.type, RDF.Statement))
g2.add((stmt2, RDF.subject,
     URIRef('http://rdflib.net/store/ConjunctiveGraph')))
g2.add((stmt2, RDF.predicate, RDF.type))
g2.add((stmt2, RDF.object, RDFS.Class))
g3.add((stmt3, RDF.type, RDF.Statement))
g3.add((stmt3, RDF.subject,
     URIRef('http://rdflib.net/store/ConjunctiveGraph')))
g3.add((stmt3, RDF.predicate, RDFS.comment))
g3.add((stmt3, RDF.object, Literal(
   "The top-level aggregate graph - The sum " +
    "of all named graphs within a Store")))
len(list(ConjunctiveGraph(store).subjects(RDF.type, RDF.Statement)))
print(list(ConjunctiveGraph(store).subjects(RDF.type, RDF.Statement)))
len(list(ReadOnlyGraphAggregate([g1,g2]).subjects(RDF.type, RDF.Statement)))
print(list(ReadOnlyGraphAggregate([g1,g2]).subjects(RDF.type, RDF.Statement)))


uniqueGraphNames = set(
     [graph.identifier for s, p, o, graph in ConjunctiveGraph(store
   ).quads((None, RDF.predicate, None))])

print(uniqueGraphNames)

unionGraph = ReadOnlyGraphAggregate([g1, g2])

uniqueGraphNames = set(
 [graph.identifier for s, p, o, graph in unionGraph.quads(
  (None, RDF.predicate, None))])

print(uniqueGraphNames)


RDFLib = Namespace('http://rdflib.net/')

RDFLib.gswyhq
#rdflib.term.URIRef('http://rdflib.net/gswyhq')

RDFLib['中文']
#rdflib.term.URIRef('http://rdflib.net/中文')


rr=ResultRow({ Variable('a'): URIRef('urn:cake') }, [Variable('a')])
rr[0]
#rdflib.term.URIRef(u'urn:cake')
rr
#(rdflib.term.URIRef(u'urn:cake'))
rr.a
#rdflib.term.URIRef(u'urn:cake')
rr[Variable('a')]
#rdflib.term.URIRef(u'urn:cake')



from rdflib.graph import Graph
from rdflib.collection import Collection
from pprint import pprint
listName = BNode()
g = Graph('IOMemory')
listItem1 = BNode()
listItem2 = BNode()
g.add((listName, RDF.first, Literal(1)))
g.add((listName, RDF.rest, listItem1))
g.add((listItem1, RDF.first, Literal(2)))
g.add((listItem1, RDF.rest, listItem2))
g.add((listItem2, RDF.rest, RDF.nil))
g.add((listItem2, RDF.first, Literal(3)))
c = Collection(g,listName)
pprint([term.n3() for term in c])



g = Graph()
statementId = BNode()
g=ConjunctiveGraph()
g.add((statementId, RDF.type, RDF.Statement))
g.add((statementId, RDF.subject,   URIRef(u'http://rdflib.net/store/ConjunctiveGraph')))
g.add((statementId, RDF.predicate, RDFS.label))
g.add((statementId, RDF.object, Literal("Conjunctive Graph")))


g=Graph()
g2=Graph()
statementId = BNode()
statementId2 = BNode()
statementId3 = BNode()
g.add((statementId,RDFLib.label,RDFLib.cold))
g.add((statementId,RDFLib.type,RDFLib.Disease))
g.add((statementId2,RDFLib.type,RDFLib.Medicine))
g2.add((statementId3,RDFLib.type,RDFLib.Price))


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
def topList(node,g,predicate=RDF.rest):
    '''根据objects:node 和 predicate:RDF.rest,查找subjects'''
    for s in g.subjects(predicate,node):
        yield s

def reverseList(node,g):
    for f in g.objects(node,RDF.first):
        print(f)
    for s in g.subjects(RDF.rest,node):
        yield s

#transitiveClosure，递归查找，将参数一（函数）调用参数二，返回值供自己调用
print([rt for rt in g.transitiveClosure(topList,RDF.nil)])
for s,p,o in g:
    print(s,p,o ,sep='      ')

print([rt for rt in g.transitiveClosure(reverseList,RDF.nil)])

'''
import rdflib
g=rdflib.Graph()
g.parse('/home/gswyhq/百科/RDF/百度百科义项.rdf')

qres = g.query(
    """SELECT DISTINCT ?aname ?bname
       WHERE {
          ?a foaf:knows ?b .
          ?a foaf:name ?aname .
          ?b foaf:name ?bname .
       }""")

for row in qres:
    print("%s knows %s" % row)

q = prepareQuery(
        'SELECT ?s WHERE { ?person foaf:knows ?s .}',
        initNs = { "foaf": FOAF })

'''

def main():
    pass

if __name__ == "__main__":
    main()

