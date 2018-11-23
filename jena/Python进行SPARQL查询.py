#!/usr/bin/python3
# coding: utf-8

from SPARQLWrapper import SPARQLWrapper, JSON

# sparql = SPARQLWrapper("http://dbpedia.org/sparql")
sparql = SPARQLWrapper("http://192.168.3.145:3030/tdb_drug_new/query")
# sparql = SPARQLWrapper("http://192.168.3.145:3030/tdb/query")
sparql.setCredentials('admin', 'gswyhq')
# '''
#
#     PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
#     SELECT ?s
#     WHERE { ?s ?p ?o .}
#     LIMIT 3
#
# '''
sparql.setQuery(
    """ 
    SELECT ?subject ?predicate ?object
    WHERE {
      ?subject ?predicate ?object
    }
    LIMIT 25
    """
)
sparql.setReturnFormat(JSON)
results = sparql.query().convert()


for result in results["results"]["bindings"]:
    print(result)
    # print(result["s"]["value"])

def main():
    pass


if __name__ == '__main__':
    main()