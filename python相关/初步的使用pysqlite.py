#!/usr/bin/python
# -*- coding:utf8 -*- #
from __future__ import  generators
from __future__ import  division
from __future__ import  print_function
from __future__ import  unicode_literals
import sys,os,json

if sys.version >= '3':
    PY3 = True
else:
    PY3 = False

if PY3:
    import pickle
else:
    import cPickle as pickle
    from codecs import open

# http://kodango.com/pysqlite-tutorial

1. 首先导入sqlite3模块

import sqlite3
2. 接着创建数据库链接

conn = sqlite3.connect('test.db')

其中"test.db"是数据库的名称，如果数据库文件不存在，就会自动创建；否则，就打开指定的数据库文件，同时创建一个数据库连接对象，该对象主要有以下操作：

commit(): 事务提交
rollback(): 事务回滚
close(): 关闭一个数据库连接
cursor(): 创建一个游标
其中，commit()方法用于提交事务，rollback()方法用于回滚至上次调用commit()方法的地方。可以通过Connection.isolation_level定义事务隔离级别，当该属性设置成None时，它会自动提交事务，不需要显式地调用commit()方法。

除了直接指定数据库文件之外，还有一种方法是在内存中创建数据库。方法是将":memory:"作为参数传递给sqlite.connect()函数：

conn = sqlite3.connect(":memory:")
3. 接下来就需要创建游标对象

cur = conn.cursor()


游标提供了一种对从表中检索出的数据进行操作的灵活手段，就本质而言，游标实际上是一种能从包括多条数据记录的结果集中每次提取一条记录的机制。游标总是与一条SQL 选择语句相关联。因为游标由结果集（可以是零条、一条或由相关的选择语句检索出的多条记录）和结果集中指向特定记录的游标位置组成。当决定对结果集进行处理时，必须声明一个指向该结果集的游标。

游标对象主要包含以下方法：

execute(): 执行sql语句
executemany(): 执行多条sql语句
close(): 关闭游标
fetchone(): 从结果中取一条记录
fetchmany(): 从结果中取多条记录
fetchall(): 从结果中取出所有记录



import sqlite3

persons = [
    ("Hugo", "Boss"),
    ("Calvin", "Klein")
    ]

con = sqlite3.connect(":memory:")

# Create the table
con.execute("create table person(firstname, lastname)")

# Fill the table （这里使用PySqlite提供的占用符格式，提高安全性）
con.executemany("insert into person(firstname, lastname) values (?, ?)", persons)

# Print the table contents （使用迭代的方法获取查询结果）
# con.execute(..)方法返回游标对象，避免手动创建游标对象。
for row in con.execute("select firstname, lastname from person"):
    print row

print "I just deleted", con.execute("delete from person").rowcount, "rows"



# http://gashero.yeax.com/?p=13

例子1

这个例子显示了一个打印 people 表格内容的最简单的例子:

from pysqlite2 import dbapi2 as sqlite
# 创建数据库连接到文件"mydb"
con=sqlite.connect("mydb")
# 获取游标对象
cur=con.cursor()
# 执行SELECT语句
cur.execute("SELECT * FROM people ORDER BY age")
# 获取所有行并显示
print cur.fetchall()
输出:

[(u'Putin', 51), (u'Yeltsin', 72)]
例子2

如下是另一个小例子展示了如何单行显示记录:

from pysqlite2 import dbapi2 as sqlite
con=sqlite.connect("mydb")
cur=con.cursor()
SELECT="SELECT name_last,age FROM people ORDER BY age, name_last"
# 1. 第一种显示记录的方法
cur.execute(SELECT)
for (name_last,age) in cur:
    print '%s is %d years old.'%(name_last,age)
# 2. 第二种显示记录的方法
cur.execute(SELECT)
for row in cur:
    print '%s is %d years old.'%(row[0], row[1])
输出:

Putin is 51 years old.
Yeltsin is 72 years old.
Putin is 51 years old.
Yeltsin is 72 years old.
例子3

如下的程序以表格的方式打印表格内容:

from pysqlite2 import dbapi2 as sqlite
FIELD_MAX_WIDTH=20
TABLE_NAME='people'
SELECT="SELECT * FROM %s ORDER BY age,name_last"%TABLE_NAME
con=sqlite.connect("mydb")
cur=con.cursor()
cur.execute(SELECT)
#打印表头
for fieldDesc in cur.description:
    print fieldDesc[0].ljust(FIELD_MAX_WIDTH),
print #结束表头行
print '-'*78

#每行打印一次值
fieldIndices=range(len(cur.description))
for row in cur:
    for fieldIndex in fieldIndices:
        fieldValue=str(row[fieldIndex])
        print fieldValue.ljust(FIELD_MAX_WIDTH),
    print
输出:

name_last               age
---------------------------------------------
Putin                   51
Yeltsin                 72
例子4

插入人员信息到 people 表格:

from pysqlite2 import dbapi2 as sqlite
con=sqlite.connect("mydb")
cur=con.cursor()
newPeople=(
    ('Lebed',53),
    ('Zhirinovsky',57),
)
for person in newPeople:
    cur.execute("INSERT INTO people (name_last,age) VALUES (?,?)",person)
#修改之后必须明确的提交
con.commit()
注意参数化的SQL语句。当处理重复语句时，这会更快并产生更少的错误，相对于手动生成SQL语句。

而上面的核心语句:

for person in newPeople:
    cur.execute("INSERT INTO people (name_last,age) VALUES (?,?)",person)
可以被重写为:

cur.executemany("INSERT INTO people (name_last,age) VALUES (?,?)",newPeople)



def main():
    pass


if __name__ == "__main__":
    main()
