#!/usr/bin/python3
# coding: utf-8

# gswyhq@gswyhq-PC:~$ sudo pip3 install pyorient
import pyorient
# 1、创建客户端实例意味着创建连接。
client = pyorient.OrientDB("localhost", 2424)
session_id = client.connect("root", "gswyhq")

# 2、创建名为DB_Demo的数据库。
db_name = 'abcde'
client.db_create(db_name, pyorient.DB_TYPE_GRAPH, pyorient.STORAGE_TYPE_MEMORY)

# 3、打开名为DB_Demo的DB。
client.db_open(db_name, "admin", "admin")

# 4、创建类my_class。
cluster_id = client.command("create class my_class extends V")

# 5、创建属性id和名称。
cluster_id = client.command("create property my_class.id Integer")
cluster_id = client.command("create property my_class.name String")

# 6、将记录插入我的类。
client.command("insert into my_class('id','name') values( 1201, 'satish')")


cluster_id = client.command( "create class my_class extends V" )
client.command(
    "insert into my_class ( 'accommodation', 'work', 'holiday' ) values( 'B&B', 'garage', 'mountain' )"
)

# 更多使用示例见： http://orientdb.com/docs/last/PyOrient-Client-Command.html

def main():
    pass


if __name__ == '__main__':
    main()