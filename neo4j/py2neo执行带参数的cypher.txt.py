#!/usr/bin/python3
# coding: utf-8

from py2neo import Graph, Node, Relationship
from py2neo.packages.httpstream import http
http.socket_timeout = 9999

graph = Graph(host="192.168.3.145", user="neo4j", password='gswyhq', http_port=7474, bolt_port=7687, secure=False, bolt=False)
tx = graph.begin()
tx.run("CREATE (n: Label {props}) RETURN n", parameters={
      "props" : {
        "name" : "My Node测试"
      }
    })
tx.commit()

def main():
    pass


if __name__ == '__main__':
    main()