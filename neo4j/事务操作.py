#!/usr/bin/python3
# coding: utf-8

import py2neo

from py2neo import Graph, Node, Relationship


def authenticateAndConnect():
    # py2neo.authenticate('localhost:47474', 'neo4j', 'gswewf')
    return Graph(host='localhost', user='neo4j', password='gswewf', http_port=7474, bolt_port=7687)


def actorsDictionary():
    return

# match (n) where n.name='Emil' return n
# CREATE (ee:Person { name: "Emil", from: "Sweden", klout: 99 })
# MERGE (charlie:Person { name:'Charlie Sheen', age:10 })
# match (n)-[r]-(n1) where n.title='Answer' return n,r,n1

def createData():
    graph = authenticateAndConnect()
    tx = graph.begin()

    # 清空数据库
    del_cypher = 'match(n) optional match(n)-[r]-() delete n,r'
    movie = Node('Movie', title='Answer')
    cypher = 'MERGE (ee:Person { name: "Emil", from: "Sweden", klout: 99 })'
    personDictionary = [{'name': 'Dan', 'born': 2001}, {'name': 'Brown', 'born': 2001}]
    for i in range(10):
        for person in personDictionary:
            person = Node('Person', name=person['name'], born=person['born'])

            # 删除原有数据
            tx.run(del_cypher)

            print('执行cypher语句')
            tx.run(cypher)
            tx.merge(person)
            actedIn = Relationship(person, 'ACTED_IN', movie)
            tx.merge(actedIn)
        if i > 3:
            personDictionary = [{'name': 'Dan', 'born': 2001}, {'name': 'Brown', 'born!@#$%^&*()_+': 2001}]
    print('提交')
    tx.commit()



def main():

    # 冲突提交事务请求
    for i in range(10):
        createData()


if __name__ == '__main__':
    main()
