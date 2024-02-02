#!/usr/bin/env python
# coding=utf-8

# pip3 install oracledb

# 独立链接模式
# 一般方法：

import oracledb
import getpass

userpwd = getpass.getpass("Enter password: ")

connection = oracledb.connect(user="hr", password=userpwd,
                              dsn="dbhost.example.com/orclpdb")

# 其他功能代码

# 关闭连接
connection.close()

######################################################################################################
# 或者可以使用以下方法：

username="hr"
userpwd = os.environ.get("PYTHON_PASSWORD")
host = "localhost"
port = 1521
service_name = "orclpdb"

dsn = f'{username}/{userpwd}@{port}:{host}/{service_name}'
connection = oracledb.connect(dsn)

# 其他功能代码

# 关闭连接
connection.close()

######################################################################################################
# 使用连接池的方法
# 初始化连接
pool = oracledb.create_pool(user="test", password='test', dsn="192.168.3.105:21521/orclpdb1.localdomain",
                            min=2, max=5, increment=1)

# Acquire 连接到池
connection = pool.acquire()

# 使用连接池
his_schemas_list = []
with connection.cursor() as cursor:
    for result in cursor.execute("SELECT username FROM dba_users"):
        print(result)
        his_schemas_list.append(result[0])

schemas_list = [ 'GCCIWSDATA',
 'GCCJOB',
 'GENESYSDATA']

for schemas in schemas_list:
    if schemas in his_schemas_list:
        continue
    # 创建用户(模式), 这里test 为设置的密码password
    sql1 = f"create user {schemas} identified by test"

    # 赋予角色
    sql2 = f"GRANT CONNECT, RESOURCE, DBA TO {schemas}"

    # 切换用户
    sql3 = f"CONNECT {schemas}/test@ORCLPDB1"

    # 创建表：
    sql4 = f"""CREATE TABLE {schemas}.table1 (
        created_by varchar(100) ,
        created_date varchar(100) ,
        updated_date varchar(100) ,
        updated_by varchar(100)
    )"""

    # python包oracledb使用的时候，SQL末尾不要有分号；

# 释放连接池
pool.release(connection)

# 关闭连接池
pool.close()
######################################################################################################


def main():
    pass


if __name__ == "__main__":
    main()
