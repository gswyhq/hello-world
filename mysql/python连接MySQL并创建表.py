#!/usr/bin/python3
# encoding:utf8

import pymysql.cursors
from pymysql.err import IntegrityError
from conf.mysql_db import MYSQL_PASSWORD, MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_DB, NEW_QUESTION_MYSQL_DB,NEW_QUESTION_MYSQL_HOST, NEW_QUESTION_MYSQL_PASSWORD, NEW_QUESTION_MYSQL_USER
from logger.logger import logger

class ConnMysql():
    def __init__(self, host='', user='', password='', db='', port=3306):
        logger.info("链接数据库： {}".format((host, port, db, user)))
        self.__host = host
        self.__port = int(port)
        self.__user = user
        self.__db = db
        self.__password = password
        self.reconnect()
        # self.db.escape()
        # 插入(查询)数据时遇到一些特殊字符会使得程序中断。操作失败。比如 \这样的转义字符
        # 解决方案:插入(查询)之前用connection.escape(str)处理一下即可
        # 代码示例
        # sql_pattern = "select * from my_collection where name = %s"  # 注意，这里直接用%s,不要给%s加引号，因为后面转移过后会自动加引号
        # name = "xxx\xxx"
        # name = connection.escape(name)

    @property
    def get_host(self):
        return (self.__host, self.__port)

    def reconnect(self):
        """断开之后自动重连"""
        if self.__host:
            self.db = pymysql.connect(host=self.__host,
                                  port=self.__port,
                                  user=self.__user,
                                  password=self.__password,
                                  db=self.__db,
                                  charset='utf8mb4',  # utf8mb4
                                  cursorclass=pymysql.cursors.DictCursor)
            self.db.ping()
        else:
            self.db = None

    def executemany(self):
        from datetime import date
        data = [
            ('Jane', date(2005, 2, 12)),
            ('Joe', date(2006, 5, 23)),
            ('John', date(2010, 10, 3)),
        ]
        stmt = "INSERT INTO employees (first_name, hire_date) VALUES (%s, %s)"
        with self.db.cursor() as cursor:
            cursor.executemany(stmt, data)
            self.db.commit()
        sql = """
        INSERT INTO employees (first_name, hire_date)VALUES
                                ('Jane', '2005-02-12'),
                                ('Joe', '2006-05-23'),
                                ('John', '2010-10-03')
        """

    def multiple_rows(self, datas):
        # 返回可用于multiple rows的sql拼装值
        def multipleRows(params):
            ret = []
            # 根据不同值类型分别进行sql语法拼装
            for param in params:
                if isinstance(param, (int, float, bool)):
                    ret.append(str(param))
                elif isinstance(param, str):
                    ret.append('"' + param + '"')
                else:
                    print('unsupport value: %s ' % param)
            return '(' + ','.join(ret) + ')'

        sql = "INSERT INTO mtable(field1, field2, field3) VALUES (%s, %s, %s)"
        counts = 0
        with self.db.cursor() as cur:
            for item in datas:
                v1, v2, v3 = item
                batch_list.append(multipleRows([v1, v2, v3]))
                # 批量插入
                if len(batch_list) == self.db.mysql_batch_num:
                    sql = "INSERT INTO mtable(field1, field2, field3) VALUES %s " % ','.join(batch_list)
                    cur.execute(sql)
                    self.db.commit()
                    batch_list = []
                    counts += len(batch_list)
                    self.db.print_log("inserted:" + str(counts))


    def __del__(self):
        if self.__host:
            self.db.close()

# conn_mysql = ConnMysql(host=MYSQL_HOST, user=MYSQL_USER, password=MYSQL_PASSWORD, db=MYSQL_DB)
new_question_mysql = ConnMysql(host=NEW_QUESTION_MYSQL_HOST, port=MYSQL_PORT, user=NEW_QUESTION_MYSQL_USER, password=NEW_QUESTION_MYSQL_PASSWORD, db=NEW_QUESTION_MYSQL_DB)

def test_cr():
    db = pymysql.connect(host=MYSQL_HOST,
                         user=MYSQL_USER,
                         password=MYSQL_PASSWORD,
                         db=MYSQL_DB,
                         charset='utf8',  # utf8mb4
                         cursorclass=pymysql.cursors.DictCursor)
    table_name = 'yhb_qa'
    sql2 = 'INSERT INTO yhb_qa( qa_id, create_time, uid, tag1, answer, modify_time, question, similar_question, tag2) VALUES ( "123", NOW(), "123.12.8.12", "资讯类", "中国人", NOW(), "你是哪里人", "", "投诉")'

    create_table_sql = '''CREATE TABLE yhb_qa(
                            id int(10) AUTO_INCREMENT PRIMARY KEY COMMENT '唯一自增id',
                            qa_id int(10) NOT NULL  COMMENT '标准问题对应的id',
                            uid VARCHAR(20) COMMENT '用户id',
                            question varchar(100) NOT NULL unique COMMENT '问题',
                            answer text NOT NULL COMMENT '答案',
                            similar_question text COMMENT '相似问题',
                            tag1 tinytext COMMENT '一级问题分类',
                            tag2 tinytext COMMENT '二级问题分类',
                            create_time DATETIME NOT NULL COMMENT '创建时间',
                            modify_time TIMESTAMP COMMENT '修改时间',
                            UNIQUE  INDEX (question)
                            )DEFAULT CHARSET=utf8;'''
    with db.cursor() as cursor:
        # 使用 execute() 方法执行 SQL，如果表存在则删除
        cursor.execute("DROP TABLE IF EXISTS {}".format(table_name))
        print("create_table_sql: {}".format(create_table_sql))
        cursor.execute(create_table_sql)
        db.commit()
        print("sql:{}".format(sql2))
        try:
            cursor.execute(sql2)
            db.commit()
        except IntegrityError:
            print("数据已存在")

def main():
    test_cr()

if __name__ == '__main__':
    main()

# # 连接到数据库
# connection = pymysql.connect(host='localhost',
#                              user='user',
#                              password='passwd',
#                              db='db',
#                              charset='utf8mb4',
#                              cursorclass=pymysql.cursors.DictCursor)
#
# try:
#     with connection.cursor() as cursor:
#         # 添加一个记录
#         sql = "INSERT INTO `users` (`email`, `password`) VALUES (%s, %s)"
#         cursor.execute(sql, ('webmaster@python.org', 'very-secret'))
#
#     # connection 不会自动提交. 因此需要显式commit
#     connection.commit()
#
#     with connection.cursor() as cursor:
#         # 读取一条记录
#         sql = "SELECT `id`, `password` FROM `users` WHERE `email`=%s"
#         cursor.execute(sql, ('webmaster@python.org',))
#         result = cursor.fetchone()
#         print(result)
# finally:
#     connection.close()


# In[175]: sql3 = '''CREATE TABLE class_member(
# id TINYINT(2) AUTO_INCREMENT PRIMARY KEY,
# name VARCHAR(20) NOT NULL UNIQUE,
# age TINYINT(2) NOT NULL,
# create_time DATETIME NOT NULL,
# modify_time TIMESTAMP
# );'''
# In[176]: cursor.execute(sql3)
# Out[176]: 0
#     In[255]: cursor.execute('INSERT INTO class_member(name,age,create_time) VALUES ("jack",24,NOW())')
#     Out[255]: 1
#     In[258]: cursor.execute('INSERT INTO class_member(name,age,create_time) VALUES ( "lily",25,NOW())')
#     Out[258]: 1
#     In[262]: cursor.execute('INSERT INTO class_member(name,age,create_time) VALUES ("lucy",25,NOW())')
#     Out[262]: 1
#     In[266]: cursor.execute('UPDATE class_member SET age=25 WHERE name="jack"')
#     Out[266]: 1


# cursor.execute('create table dj1 (a char(1), b TIMESTAMP(6) )')
# cursor.execute('insert into dj1 values (1,null)')  # b字段自动补全插入时间
# In[201]: cursor.execute('update dj1 set a=9 where a=1; ')  # 修改后，b字段的时间也会更新
# In[172]: cursor.execute('CREATE TABLE python_tstamps ( a char(1), ts TIMESTAMP(6) )')
#     In[230]: cursor.execute('create table dj2 (a char(1),b TIMESTAMP(6) NOT NULL DEFAULT 20170329)')
#     Out[230]: 0
#     In[232]: cursor.execute('create table dj2 (a char(1),b TIMESTAMP(6) NOT NULL DEFAULT 20170329, c  timestamp(6) NOT NULL DEFAULT 20170330 )')
#     Out[232]: 0
#     In[239]: cursor.execute('create table dj3 (a char(1),b TIMESTAMP(6) , c  timestamp(6) NOT NULL DEFAULT 20170330 )')
#     Out[239]: 0
    # 字段类型为TIMESTAMP时，若设置了默认值，则数据有更新时，其值并不变动；当不设置默认值是，数据有变动，则也会变动；并且不设置默认值，需在设置默认值之前设置
    # 简言之：有个属性ON UPDATE CURRENT_TIMESTAMP，导致更新数据时，即便未涉及到该列，该列数据也被自动更新。
    # '1.MySQL默认表的第一个timestamp字段为NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP属性。
    # 2.MySQL只允许一个timestamp字段拥有[DEFAULT CURRENT_TIMESTAMP |ON UPDATE CURRENT_TIMESTAMP]属性'
# cursor.execute('create table dj1 (a char(1), b TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP , c TIMESTAMP(6) DEFAULT CURRENT_TIMESTAMP )')