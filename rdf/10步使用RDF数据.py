#encoding=utf-8

#http://gromgull.net/blog/2011/01/a-quick-and-dirty-guide-to-your-first-time-with-rdf/


#sudo apt-get install python-rdflib python-bsddb3
# We will use the schema file from:
# http://source.data.gov.uk/data/research/bis-research-explorer/2010-03-04/research-schema.rdf
# and the education data from:
# http://source.data.gov.uk/data/education/bis-research-explorer/2010-03-04/education.data.gov.uk.nt


#Semantic web linked data python


import rdflib


#加载数据
g=rdflib.Graph('Sleepycat')
g.open("db")

g.load("http://source.data.gov.uk/data/education/bis-research-explorer/2010-03-04/education.data.gov.uk.nt", format='nt')
g.load("http://source.data.gov.uk/data/research/bis-research-explorer/2010-03-04/research-schema.rdf")

g.close()


g=rdflib.Graph('Sleepycat')
g.open("db")

print "------------------"
print # triples:"
print len(g)

print "------------------"
print "All types:"
for x in set(g.objects(None, rdflib.RDF.RDFNS["type"])): 
    print x

print "------------------"
print "All Institutions:" 
for x in list(g.subjects(rdflib.RDF.RDFNS["type"], rdflib.URIRef('http://purl.org/vocab/aiiso/schema#Institution')))[:10]:
    print x

print "------------------"
print "Triples about UCL:" 

for t in g.triples((rdflib.URIRef('http://education.data.gov.uk/id/institution/UniversityColledgeOfLondon'), None, None)): 
    print map(str,t)

PREFIX="""
PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX foaf: <http://xmlns.com/foaf/0.1/>
PREFIX p: <http://research.data.gov.uk/def/project/>
PREFIX aiiso: <http://purl.org/vocab/aiiso/schema#>
PREFIX geo: <http://www.w3.org/2003/01/geo/wgs84_pos#>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
"""

print "------------------"
print "SPARQL Results 1:" 

for x in list(g.query(PREFIX+"SELECT ?x ?label WHERE { ?x rdfs:label ?label ; a aiiso:Institution . } LIMIT 10 ")): 
    print x


print "------------------"
print "SPARQL Results 2:" 


r=list(g.query(PREFIX+"""SELECT DISTINCT ?x ?xlabel ?y ?ylabel WHERE {
   ?x rdfs:label ?xlabel ;
      a aiiso:Institution ;
      p:organisationSize 'Public Sector' ;
      p:project ?p . 

   ?y rdfs:label ?ylabel ;
      a aiiso:Institution ;
      p:organisationSize 'Public Sector' ;
      p:project ?p .

   FILTER (?x != ?y) } LIMIT 10 """))

for x in r[:3]: 
    print map(str,x)


'''
>>> from rdflib import Graph, URIRef
INFO:rdflib:RDFLib Version: 4.2.1
>>> g = Graph()
>>> g.parse("f:/狗尾草/research-schema.rdf")
<Graph identifier=N6ffb11f67c51406394c6f2e9c18c18a2 (<class 'rdflib.graph.Graph'>)>
>>> len(g)
186
>>> for stmt in g.subject_objects("f:/狗尾草/research-schema.rdf"):
     print (stmt)

     
>>> for stmt in g.subject_objects(URIRef("http://dbpedia.org/ontology/birthDate")):
     print (stmt)

     
>>> g.parse("http://dbpedia.org/resource/Elvis_Presley")
<Graph identifier=N6ffb11f67c51406394c6f2e9c18c18a2 (<class 'rdflib.graph.Graph'>)>
>>> len(g)
2034
>>> for stmt in g.subject_objects(URIRef("http://dbpedia.org/ontology/birthDate")):
     print (stmt)

     
(rdflib.term.URIRef('http://dbpedia.org/resource/Elvis_Presley'), rdflib.term.Literal('1935-01-08', datatype=rdflib.term.URIRef('http://www.w3.org/2001/XMLSchema#date')))
>>> for stmt in g.subject_objects(URIRef("http://dbpedia.org/ontology/birthDate")):
    print ("the person represented by", str(stmt[0]), "was born on", str(stmt[1]))

    
the person represented by http://dbpedia.org/resource/Elvis_Presley was born on 1935-01-08
>>> g.parse("http://dbpedia.org/resource/Tim_Berners-Lee")
<Graph identifier=N6ffb11f67c51406394c6f2e9c18c18a2 (<class 'rdflib.graph.Graph'>)>
>>> g.parse("http://dbpedia.org/resource/Albert_Einstein")
<Graph identifier=N6ffb11f67c51406394c6f2e9c18c18a2 (<class 'rdflib.graph.Graph'>)>
>>> g.parse("http://dbpedia.org/resource/Margaret_Thatcher")
<Graph identifier=N6ffb11f67c51406394c6f2e9c18c18a2 (<class 'rdflib.graph.Graph'>)>
>>> for stmt in g.subject_objects(URIRef("http://dbpedia.org/ontology/birthDate")):
     print( "the person represented by", str(stmt[0]), "was born on", str(stmt[1]))

     
the person represented by http://dbpedia.org/resource/Tim_Berners-Lee was born on 1955-06-08
the person represented by http://dbpedia.org/resource/Albert_Einstein was born on 1879-03-14
the person represented by http://dbpedia.org/resource/Elvis_Presley was born on 1935-01-08
the person represented by http://dbpedia.org/resource/Margaret_Thatcher was born on 1925-10-13
>>>



>>> import networkx as nx
>>> G=nx.Graph()
>>> G.add_node("spam")
>>> G.add_edge(1,2)
>>> print(G.nodes())
[1, 2, 'spam']
>>> print(G.edges())
[(1, 2)]



block = ''<?xml version="1.0"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">

</rdf:RDF>''
Graph g;
g.parse( StringIO.StringIO(block), format='xml')
Update: Based on lawlesst's answer this can be simplified without the StringIO as just:

g.parse( data=block, format='xml' )
#https://github.com/RDFLib/rdflib/blob/4.1.2/rdflib/graph.py#L209



'''


