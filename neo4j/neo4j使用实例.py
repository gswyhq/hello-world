#!/usr/bin/python3
# -*- coding:utf8 -*- #

from py2neo import Graph, Node, Relationship, authenticate, NodeSelector

# 一、连接Neo4j数据库
test_graph = Graph(
		"http://localhost:7474",
		username="neo4j",
		password="neo4j"
)
# test_graph就是我们建立好的Neo4j的连接。

# 等价于：
authenticate("localhost:7474", "neo4j", "neo4j")
test_graph = Graph()

# 二、节点的建立

# 节点的建立要用到py2neo.Node，建立节点的时候要定义它的节点类型（label）以及一个基本属性（property，包括property_key和property_value）。以下代码为建立了两个测试节点。

test_node_1 = Node(label="Person", name="test_node_1")
test_node_2 = Node(label="Person", name="test_node_2")
test_graph.create(test_node_1)
test_graph.create(test_node_2)
# 这两个节点的类型(label)都是Person，而且都具有属性（property_key）为name，属性值（property_value）分别为"test_node_1"，"test_node_2"。


# 三、节点间关系的建立

# 节点间的关系（Relationship）是有向的，所以在建立关系的时候，必须定义一个起始节点和一个结束节点。值得注意的是，起始节点可以和结束节点是同一个点，这时候的关系就是这个点指向它自己。

node_1_call_node_2 = Relationship(test_node_1, 'CALL', test_node_2)
node_1_call_node_2['count'] = 1
node_2_call_node_1 = Relationship(test_node_2, 'CALL', test_node_1)
node_2_call_node_1['count'] = 2
test_graph.create(node_1_call_node_2)
test_graph.create(node_2_call_node_1)
# 如以上代码，分别建立了test_node_1指向test_node_2和test_node_2指向test_node_1两条关系，关系的类型为"CALL"，两条关系都有属性count，且值为1。

# 在这里有必要提一下，如果建立关系的时候，起始节点或者结束节点不存在，则在建立关系的同时建立这个节点。

# 四、节点/关系的属性赋值以及属性值的更新

# 节点和关系的属性初始赋值在前面节点和关系的建立的时候已经有了相应的代码，在这里主要讲述一下怎么更新一个节点/关系的属性值。

# 我们以关系建立里的 node_1_call_node_2 为例，让它的count加1，再更新到图数据库里面。

node_1_call_node_2['count'] += 1
test_graph.push(node_1_call_node_2)
# 更新属性值就使用push函数来进行更新即可。

# 五、通过属性值来查找节点和关系（find,find_one）

# 通过find和find_one函数，可以根据类型和属性、属性值来查找节点和关系。

find_code_1 = test_graph.find_one(
		label="Person",
		property_key="name",
		property_value="test_node_1"
)
print(find_code_1['name'])
# find和find_one的区别在于：
# find_one的返回结果是一个具体的节点/关系，可以直接查看它的属性和值。如果没有这个节点/关系，返回None。
# find查找的结果是一个游标，可以通过循环取到所找到的所有节点/关系。

# Graph.find_one 不推荐使用, 改用 NodeSelector，使用示例如下：
# | >> > selector = NodeSelector(test_graph)
# | >> > selected = selector.select("Person", name="Keanu Reeves")
# | >> > list(selected)
# | [(f9726ea:Person{born: 1964, name: "Keanu Reeves"})]

# | >> > selected = selector.select("Person").where("_.name =~ 'J.*'", "1960 <= _.born < 1970")
# | >> > list(selected)
# | [(a03f6eb:Person{born: 1967, name: "James Marshall"}),
# | (e59993d:Person{born: 1966, name: "John Cusack"}),
# | (c44901e:Person{born: 1960, name: "John Goodman"}),
# | (b141775:Person{born: 1965, name: "John C. Reilly"}),
# | (e40244b:Person{born: 1967, name: "Julia Roberts"})]
# 六、通过节点/关系查找相关联的节点/关系

# 如果已经确定了一个节点或者关系，想找到和它相关的关系和节点，就可以使用match和match_one。

find_relationship = test_graph.match_one(start_node=find_code_1, end_node=find_code_2, bidirectional=False)
print(find_relationship)
# 如以上代码所示，match和match_one的参数包括start_node,Relationship，end_node中的至少一个。

# bidirectional参数的意义是指关系是否可以双向。
# 如果为False,则起始节点必须为start_node，结束节点必须为end_node。如果有Relationship参数，则一定按照Relationship对应的方向。
# 如果为True，则不需要关心方向问题，会把两个方向的数据都返回。

match_relation = test_graph.match(start_node=find_code_1, bidirectional=True)
for i in match_relation:
	print(i)
	i['count'] += 1
	test_graph.push(i)
# 如以上代码所示，查找和find_code_1相关的关系。
# match里面的参数只写了start_node，bidirectional的值为True，则不会考虑方向问题，返回的是以find_code_1为起始节点和结束节点的所有关联关系。
# 如果,bidirectional的值为False，则只会返回以find_code_1为起始节点的所有关联关系。

# 七、关于节点和关系的建立

# 建立节点和关系之前最好先查找一下是否已经存在这个节点了。如果已经存在的话，则建立关系的时候使用自己查找到的这个节点，而不要新建，否则会出现一个新的节点。
# 如果一条条建立关系和节点的话，速度并不快。如果条件合适的话，最好还是用Neo4j的批量导入功能。

# 八、关于索引

# 在Neo4j 2.0版本以后，尽量使用schema index，而不要使用旧版本的索引。
# 最好在插入数据之前就建立好索引，否则索引的建立会很消耗时间。
# 索引建立是非常有必要的，一个索引可以很大程度上降低大规模数据的查询速度。

# 九、关于Neo4j应当插入的数据内容

# 我们在使用图数据库的时候必须要明确一点，图数据库能够帮助我们的是以尽量快的速度找出来不同的节点之间的关系。因此向一个节点或者关系里面插入很多其余无关的数据是完全没有必要的，会很大程度浪费硬盘资源，在检索的时候也会消耗更多的时间。

以上来源于：http://www.jianshu.com/p/a2497a33390f

# 其它实例：
# http://py2neo.org/v3/types.html
In [53]: a = Node("Person", name="Alice")

In [54]: b = Node("Person", name="Bob")

In [55]: ab = Relationship(a, "KNOWS", b)

In [56]: ab
Out[56]: (alice)-[:KNOWS]->(bob)

In [57]: c = Node("Person", name="Carol")

In [58]: from py2neo import Rela
Relatable     Relationship

In [58]: from py2neo import Relationship

In [59]: class WorksWith(Relationship): pass

In [60]: ac = WorksWith(a, c)

In [61]: ac.type()
Out[61]: 'WORKS_WITH'

In [62]: s = ab | ac

In [63]: s
Out[63]: ({(alice:Person {name:"Alice"}), (carol:Person {name:"Carol"}), (bob:Person {name:"Bob"})}, {(alice)-[:WORKS_WITH]->(carol), (alice)-[:KNOWS]->(bob)})

In [64]: s.nodes()
Out[64]:
frozenset({(alice:Person {name:"Alice"}),
           (carol:Person {name:"Carol"}),
           (bob:Person {name:"Bob"})})

In [65]: s.relationships()
Out[65]: frozenset({(alice)-[:WORKS_WITH]->(carol), (alice)-[:KNOWS]->(bob)})

In [66]: w = ab + Relationship(b, "LIKES", c) + ac

In [67]: w
Out[67]: (alice)-[:KNOWS]->(bob)-[:LIKES]->(carol)<-[:WORKS_WITH]-(alice)

# http://www.tuicool.com/articles/JfQVRje
# https://github.com/nigelsmall/py2neo/tree/v3/demo/moviegraph


from py2neo.ogm import GraphObject, Property, RelatedTo, RelatedFrom


class Movie(GraphObject):
    __primarykey__ = "title"

    title = Property()
    tagline = Property()
    released = Property()

    actors = RelatedFrom("Person", "ACTED_IN")
    directors = RelatedFrom("Person", "DIRECTED")
    producers = RelatedFrom("Person", "PRODUCED")
    comments = RelatedTo("Comment", "COMMENT")

    def __lt__(self, other):
        return self.title < other.title


class Person(GraphObject):
    # 定义一个__primarykey__。这告诉py2neo哪个属性应该被视为push和pull操作的唯一标识符。
    # 我们也可以定义一个__primarylabel__，尽管默认情况下，将使用类名Person
    __primarykey__ = "name"

    # 定义一个Person类有两个properties.
    # Properties在Neo4j没有固定类型
    name = Property()
    born = Property()

    # 引入两个新的属性类型：RelatedTo和RelatedFrom。这些定义了以相似方式连接的相关对象的集合。
    acted_in = RelatedTo(Movie)  # 描述了一组相关的电影节点，它们都通过传出ACTED_IN关系连接；关系类型默认为匹配属性名本身
    directed = RelatedTo(Movie)
    produced = RelatedTo(Movie)

    def __lt__(self, other):
        return self.name < other.name


class Comment(GraphObject):

    date = Property()
    name = Property()
    text = Property()

    subject = RelatedFrom(Movie, "COMMENT")

    def __init__(self, date, name, text):
        self.date = date
        self.name = name
        self.text = text

    def __lt__(self, other):
        return self.date < other.date

def test():
    # 实例
    # 我们要从数据库中提取Keanu Reeves，并把他连接到Bill & Ted's Excellent Adventure。
    # 首先我们需要使用GraphObject类方法选择actor，
    # 通过Person子类来选择。
    # 然后，我们可以创建一个新的电影对象，添加这个电影acted_in由有Reeves集，最终把一切回到图表。
    keanu = Person.select(graph, "Keanu Reeves").first()
    bill_and_ted = Movie()
    bill_and_ted.title = "Bill & Ted's Excellent Adventure"
    keanu.acted_in.add(bill_and_ted)
    graph.push(keanu)

    # 要输出名称以“K”开头的每个actor的名称：
    for person in Person.select(graph).where("_.name =~ 'K.*'"):
        # 下划线字符在这里用于指代被匹配的一个或多个节点
        print(person.name)



更多详细说明：
克隆到本地
gswewf@gswewf-pc:~$ git clone https://github.com/nigelsmall/py2neo.git
gswewf@gswewf-pc:~/py2neo$ cd book/
生成html说明文档
gswewf@gswewf-pc:~/py2neo/book$ make html
gswewf@gswewf-pc:~/py2neo/book$ cd _build/html/
gswewf@gswewf-pc:~/py2neo/book/_build/html$ ls
database.html  genindex.html  _modules     objects.inv  py-modindex.html  searchindex.js  _static
ext            index.html     neokit.html  ogm.html     search.html       _sources        types.html

双击打开index.html 即可观看文档
