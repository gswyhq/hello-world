# -*- coding: utf-8 -*-
# 安装mysql数据库：zy@ubuntu:~$ sudo apt-get install mysql-server mysql-client
# 中间会让设置一次root用户的密码

# 安装python包：zy@ubuntu:~$ sudo pip3 install PyMySQL

# http://www.runoob.com/python3/python3-mysql.html

# 创建数据库
# 使用客户端Navicat for MySQL连接mysql数据库；
# 新建数据库：file-> New Database
# 设置数据库的名字： Database Name: yhb
# 设置数据库字符编码： Character set: utf8mb4 -- UTF-8 Unicode  # utf8mb4是utf8的超集
# 设置排序规则: Collation: utf8mb4_unicode_520_ci
# 例如:utf8_danish_ci
# ci是'case insensitive'的缩写;表示不分大小写
# 同一个character set的不同collation的区别在于排序、字符春对比的准确度（相同两个字符在不同国家的语言中的排序规则可能是不同的）以及性能。
# 例如：utf8_general_ci在排序的准确度上要差于utf8_unicode_ci， 当然，对于英语用户应该没有什么区别。但性能上（排序以及比对速度）要略优于utf8_unicode_ci.

import pymysql.cursors

# 连接到数据库
connection = pymysql.connect(host='localhost',
                             user='user',
                             password='passwd',
                             db='db',
                             charset='utf8mb4',
                             cursorclass=pymysql.cursors.DictCursor)

try:
    with connection.cursor() as cursor:
        # 添加一个记录
        sql = "INSERT INTO `users` (`email`, `password`) VALUES (%s, %s)"
        cursor.execute(sql, ('webmaster@python.org', 'very-secret'))

    # connection 不会自动提交. 因此需要显式commit
    connection.commit()

    with connection.cursor() as cursor:
        # 读取一条记录
        sql = "SELECT `id`, `password` FROM `users` WHERE `email`=%s"
        cursor.execute(sql, ('webmaster@python.org',))
        result = cursor.fetchone()
        print(result)
finally:
    connection.close()


# 示例二
# 打开数据库连接
db = pymysql.connect("localhost","testuser","test123","TESTDB" )

# 使用 cursor() 方法创建一个游标对象 cursor
cursor = db.cursor()

# 使用 execute()  方法执行 SQL 查询
cursor.execute("SELECT VERSION()")

# 使用 fetchone() 方法获取单条数据.
data = cursor.fetchone()

print ("Database version : %s " % data)

# 关闭数据库连接
db.close()


# 创建数据库表

# 打开数据库连接
db = pymysql.connect("localhost","testuser","test123","TESTDB" )

# 使用 cursor() 方法创建一个游标对象 cursor
cursor = db.cursor()

# 使用 execute() 方法执行 SQL，如果表存在则删除
cursor.execute("DROP TABLE IF EXISTS EMPLOYEE")

# 使用预处理语句创建表
sql = """CREATE TABLE EMPLOYEE (
         FIRST_NAME  CHAR(20) NOT NULL,
         LAST_NAME  CHAR(20),
         AGE INT,
         SEX CHAR(1),
         INCOME FLOAT )"""

cursor.execute(sql)

# 关闭数据库连接
db.close()




# 数据库插入操作
# 打开数据库连接
db = pymysql.connect("localhost","testuser","test123","TESTDB" )

# 使用cursor()方法获取操作游标
cursor = db.cursor()

# SQL 插入语句
sql = """INSERT INTO EMPLOYEE(FIRST_NAME,
         LAST_NAME, AGE, SEX, INCOME)
         VALUES ('Mac', 'Mohan', 20, 'M', 2000)"""

sql2 = "INSERT INTO EMPLOYEE(FIRST_NAME, \
    LAST_NAME, AGE, SEX, INCOME) \
    VALUES ('%s', '%s', '%d', '%c', '%d' )" % \
      ('Mac', 'Mohan', 20, 'M', 2000)

try:
   # 执行sql语句
   cursor.execute(sql)
   cursor.execute(sql2)
   # 提交到数据库执行
   db.commit()
except:
   # 如果发生错误则回滚
   db.rollback()

# 关闭数据库连接
db.close()


# 查询EMPLOYEE表中salary（工资）字段大于1000的所有数据
# 打开数据库连接
db = pymysql.connect("localhost","testuser","test123","TESTDB" )

# 使用cursor()方法获取操作游标
cursor = db.cursor()

# SQL 查询语句
sql = "SELECT * FROM EMPLOYEE \
       WHERE INCOME > '%d'" % (1000)
try:
   # 执行SQL语句
   cursor.execute(sql)
   # 获取所有记录列表
   results = cursor.fetchall()
   for row in results:
      fname = row[0]
      lname = row[1]
      age = row[2]
      sex = row[3]
      income = row[4]
       # 打印结果
      print ("fname=%s,lname=%s,age=%d,sex=%s,income=%d" % \
             (fname, lname, age, sex, income ))
except:
   print ("Error: unable to fecth data")

# 关闭数据库连接
db.close()


# 更新操作用于更新数据表的的数据，以下实例将 TESTDB表中的 SEX 字段全部修改为 'M'，AGE 字段递增1
# 打开数据库连接
db = pymysql.connect("localhost","testuser","test123","TESTDB" )
# 使用cursor()方法获取操作游标
cursor = db.cursor()
# SQL 更新语句
sql = "UPDATE EMPLOYEE SET AGE = AGE + 1 WHERE SEX = '%c'" % ('M')
try:
   # 执行SQL语句
   cursor.execute(sql)
   # 提交到数据库执行
   db.commit()
except:
   # 发生错误时回滚
   db.rollback()
# 关闭数据库连接
db.close()



# 示例：删除数据表 EMPLOYEE 中 AGE 大于 20 的所有数据
# 打开数据库连接
db = pymysql.connect("localhost","testuser","test123","TESTDB" )
# 使用cursor()方法获取操作游标
cursor = db.cursor()
# SQL 删除语句
sql = "DELETE FROM EMPLOYEE WHERE AGE > '%d'" % (20)
try:
   # 执行SQL语句
   cursor.execute(sql)
   # 提交修改
   db.commit()
except:
   # 发生错误时回滚
   db.rollback()
# 关闭连接
db.close()


# 示例，事务机制可以确保数据一致性。
# 事务应该具有4个属性：原子性、一致性、隔离性、持久性。这四个属性通常称为ACID特性。
# 对于支持事务的数据库， 在Python数据库编程中，当游标建立之时，就自动开始了一个隐形的数据库事务。
# commit()方法游标的所有更新操作，rollback（）方法回滚当前游标的所有操作。每一个方法都开始了一个新的事务。
# SQL删除记录语句
sql = "DELETE FROM EMPLOYEE WHERE AGE > '%d'" % (20)
try:
   # 执行SQL语句
   cursor.execute(sql)
   # 向数据库提交
   db.commit()
except:
   # 发生错误时回滚
   db.rollback()

# pymysql.Connect()参数说明
# host(str):      MySQL服务器地址
# port(int):      MySQL服务器端口号
# user(str):      用户名
# passwd(str):    密码
# db(str):        数据库名称
# charset(str):   连接编码
#
# connection对象支持的方法
# cursor()        使用该连接创建并返回游标
# commit()        提交当前事务
# rollback()      回滚当前事务
# close()         关闭连接
#
# cursor对象支持的方法
# execute(op)     执行一个数据库的查询命令
# fetchone()      取得结果集的下一行
# fetchmany(size) 获取结果集的下几行
# fetchall()      获取结果集中的所有行
# rowcount()      返回数据条数或影响行数
# close()         关闭游标对象

def create_modify_time():
    """插入记录时间及修改时间"""
    sql3 = '''CREATE TABLE class_member(
    id TINYINT(2) AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(20) NOT NULL UNIQUE,
    age TINYINT(2) NOT NULL,
    create_time DATETIME NOT NULL,
    modify_time TIMESTAMP
    );'''
    import time
    db = pymysql.connect(host=MYSQL_HOST,
                         user=MYSQL_USER,
                         password=MYSQL_PASSWORD,
                         db=MYSQL_DB,
                         charset='utf8mb4',
                         cursorclass=pymysql.cursors.DictCursor)
    cursor = db.cursor()
    cursor.execute(sql3)
    cursor.execute('INSERT INTO class_member(name,age,create_time) VALUES ("jack",24,NOW())')
    db.commit()
    time.sleep(2)
    cursor.execute('INSERT INTO class_member(name,age,create_time) VALUES ( "lily",25,NOW())')
    db.commit()
    time.sleep(2)
    cursor.execute('INSERT INTO class_member(name,age,create_time) VALUES ("lucy",25,NOW())')
    db.commit()
    time.sleep(2)
    cursor.execute('UPDATE class_member SET age=25 WHERE name="jack"')
    db.commit()
    time.sleep(2)



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

