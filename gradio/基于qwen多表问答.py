#!/usr/bin/env python
# coding=utf-8

import os
import re, json
import requests
import sqlparse
from datetime import datetime
import os
from openai import OpenAI

# 令牌来源：https://modelscope.cn/my/myaccesstoken
MODELSCOPE_SDK_TOKEN = os.getenv("MODELSCOPE_SDK_TOKEN")

if MODELSCOPE_SDK_TOKEN is None:
    MODELSCOPE_SDK_TOKEN = 'abcd7bcd-5fa2-4e78-9477-22e7f432f109'


DATA_SOURCE_DICT = {}

# 实现一个选择读取展示excel表的页面，选择下拉列表，展示表格内容
import gradio as gr
import pandas as pd
import time  # 导入time模块，虽然在这个例子中未直接使用，但可用于其他功能如延迟
import psycopg2 # pip3 install psycopg2-binary
from psycopg2 import OperationalError

# 数据库连接参数
db_params = {
    'dbname': 'postgres',
    'user': 'supersonic_user',
    'password': 'supersonic_password',
    'host': '192.168.1.106',
    'port': '15432'
}

'''
生成PostgreSQL数据库，5张表的建表语句
我们将创建一个简单的电子商务数据库，包含以下5张表：

demo_20250120_customers：客户信息
demo_20250120_orders：订单信息
demo_20250120_order_items：订单项信息
demo_20250120_products：产品信息
demo_20250120_categories：产品类别信息
建表语句

-- 创建 demo_20250120_customers 客户信息表
CREATE TABLE demo_20250120_customers (
    customer_id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    age INTEGER,
    phone VARCHAR(20),
    province VARCHAR(50),
    city VARCHAR(50),
    address TEXT
);

COMMENT ON COLUMN demo_20250120_customers.customer_id IS '客户ID';
COMMENT ON COLUMN demo_20250120_customers.name IS '姓名';
COMMENT ON COLUMN demo_20250120_customers.age IS '年龄';
COMMENT ON COLUMN demo_20250120_customers.phone IS '电话';
COMMENT ON COLUMN demo_20250120_customers.province IS '省份';
COMMENT ON COLUMN demo_20250120_customers.city IS '城市';
COMMENT ON COLUMN demo_20250120_customers.address IS '地址';

INSERT INTO demo_20250120_customers (name, age, phone, province, city, address)
VALUES ('张三', 30, '13800000001', '广东省', '深圳市', '南山区科技路1号');

INSERT INTO demo_20250120_customers (name, age, phone, province, city, address)
VALUES ('李四', 25, '13900000002', '北京市', '朝阳区', '建国门外大街1号');

INSERT INTO demo_20250120_customers (name, age, phone, province, city, address)
VALUES ('王五', 40, '13600000003', '上海市', '浦东新区', '陆家嘴环路1000号');

INSERT INTO demo_20250120_customers (name, age, phone, province, city, address)
VALUES ('赵六', 35, '13500000004', '浙江省', '杭州市', '西湖区文三路501号');

INSERT INTO demo_20250120_customers (name, age, phone, province, city, address)
VALUES ('孙七', 28, '13700000005', '江苏省', '南京市', '鼓楼区汉中路1号');

INSERT INTO demo_20250120_customers (name, age, phone, province, city, address)
VALUES ('周八', 32, '13300000006', '山东省', '青岛市', '市南区香港中路8号');

INSERT INTO demo_20250120_customers (name, age, phone, province, city, address)
VALUES ('吴九', 45, '13000000007', '湖北省', '武汉市', '江岸区解放大道1000号');

INSERT INTO demo_20250120_customers (name, age, phone, province, city, address)
VALUES ('郑十', 31, '13200000008', '湖南省', '长沙市', '岳麓区枫林一路2号');

INSERT INTO demo_20250120_customers (name, age, phone, province, city, address)
VALUES ('陈十一', 27, '13100000009', '四川省', '成都市', '武侯区人民南路四段1号');

INSERT INTO demo_20250120_customers (name, age, phone, province, city, address)
VALUES ('黄十二', 33, '15000000010', '重庆市', '渝中区', '解放碑步行街1号');

INSERT INTO demo_20250120_customers (name, age, phone, province, city, address)
VALUES ('杨十三', 29, '15100000011', '河北省', '石家庄市', '长安区和平东路217号');

INSERT INTO demo_20250120_customers (name, age, phone, province, city, address)
VALUES ('郭十四', 36, '15200000012', '山西省', '太原市', '小店区长治路226号');

INSERT INTO demo_20250120_customers (name, age, phone, province, city, address)
VALUES ('何十五', 42, '15300000013', '陕西省', '西安市', '雁塔区小寨西路121号');

INSERT INTO demo_20250120_customers (name, age, phone, province, city, address)
VALUES ('高十六', 38, '15500000014', '辽宁省', '沈阳市', '沈河区青年大街1号');

INSERT INTO demo_20250120_customers (name, age, phone, province, city, address)
VALUES ('林十七', 26, '15600000015', '福建省', '福州市', '鼓楼区五四路1号');

INSERT INTO demo_20250120_customers (name, age, phone, province, city, address)
VALUES ('许十八', 34, '15700000016', '安徽省', '合肥市', '包河区金寨路322号');

INSERT INTO demo_20250120_customers (name, age, phone, province, city, address)
VALUES ('邓十九', 41, '15800000017', '江西省', '南昌市', '东湖区八一大道333号');

INSERT INTO demo_20250120_customers (name, age, phone, province, city, address)
VALUES ('萧二十', 37, '15900000018', '河南省', '郑州市', '金水区花园路1号');

-- 创建 demo_20250120_orders 订单信息表
CREATE TABLE demo_20250120_orders (
    order_id SERIAL PRIMARY KEY,
    customer_id INT NOT NULL,
    order_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    total_amount DECIMAL(10, 2) NOT NULL
);

COMMENT ON COLUMN demo_20250120_orders.order_id IS '订单ID';
COMMENT ON COLUMN demo_20250120_orders.customer_id IS '客户ID';
COMMENT ON COLUMN demo_20250120_orders.order_date IS '下单日期';
COMMENT ON COLUMN demo_20250120_orders.total_amount IS '订单总金额';

INSERT INTO demo_20250120_orders (customer_id, order_date, total_amount) VALUES
(1, '2025-01-15 10:00:00', 150.00),
(2, '2025-01-16 11:30:00', 200.00),
(3, '2025-01-17 12:45:00', 120.00),
(4, '2025-01-18 13:00:00', 180.00),
(5, '2025-01-19 14:15:00', 220.00),
(6, '2025-01-20 15:30:00', 160.00),
(7, '2025-01-21 16:45:00', 190.00),
(8, '2025-01-22 17:00:00', 210.00),
(9, '2025-01-23 18:15:00', 170.00),
(10, '2025-01-24 19:30:00', 230.00),
(11, '2025-01-25 20:45:00', 140.00),
(12, '2025-01-26 21:00:00', 130.00),
(13, '2025-01-27 22:15:00', 110.00),
(14, '2025-01-28 23:30:00', 90.00),
(15, '2025-01-29 00:45:00', 80.00),
(16, '2025-01-30 01:00:00', 70.00),
(17, '2025-01-31 02:15:00', 60.00),
(18, '2025-02-01 03:30:00', 50.00),
(19, '2025-02-02 04:45:00', 40.00),
(20, '2025-02-03 05:00:00', 30.00);


-- 创建 demo_20250120_order_items 订单项信息表
CREATE TABLE demo_20250120_order_items (
    order_item_id SERIAL PRIMARY KEY,
    order_id INT NOT NULL,
    product_id INT NOT NULL,
    quantity INT NOT NULL,
    price DECIMAL(10, 2) NOT NULL
);

COMMENT ON COLUMN demo_20250120_order_items.order_item_id IS '订单项ID';
COMMENT ON COLUMN demo_20250120_order_items.order_id IS '订单ID';
COMMENT ON COLUMN demo_20250120_order_items.product_id IS '产品ID';
COMMENT ON COLUMN demo_20250120_order_items.quantity IS '数量';
COMMENT ON COLUMN demo_20250120_order_items.price IS '单价';

INSERT INTO demo_20250120_order_items (order_id, product_id, quantity, price) VALUES
(1, 1, 2, 50.00),
(1, 2, 1, 50.00),
(2, 3, 1, 100.00),
(3, 4, 3, 40.00),
(4, 5, 2, 90.00),
(5, 6, 1, 110.00),
(6, 7, 2, 80.00),
(7, 8, 1, 95.00),
(8, 9, 2, 105.00),
(9, 10, 1, 85.00),
(10, 11, 3, 75.00),
(11, 12, 2, 70.00),
(12, 13, 1, 65.00),
(13, 14, 2, 55.00),
(14, 15, 1, 45.00),
(15, 16, 3, 40.00),
(16, 17, 2, 35.00),
(17, 18, 1, 30.00),
(18, 19, 2, 25.00),
(19, 20, 1, 20.00),
(20, 1, 1, 50.00),
(20, 2, 1, 50.00);

-- 创建 demo_20250120_products 产品信息表
CREATE TABLE demo_20250120_products (
    product_id SERIAL PRIMARY KEY,
    product_name VARCHAR(100) NOT NULL,
    category_id INT NOT NULL,
    price DECIMAL(10, 2) NOT NULL,
    stock INT NOT NULL
);

COMMENT ON COLUMN demo_20250120_products.product_id IS '产品ID';
COMMENT ON COLUMN demo_20250120_products.product_name IS '产品名称';
COMMENT ON COLUMN demo_20250120_products.category_id IS '类别ID';
COMMENT ON COLUMN demo_20250120_products.price IS '价格';
COMMENT ON COLUMN demo_20250120_products.stock IS '库存量';


INSERT INTO demo_20250120_products (product_name, category_id, price, stock)
VALUES ('苹果电脑', 1, 9999.00, 50);

INSERT INTO demo_20250120_products (product_name, category_id, price, stock)
VALUES ('华为手机', 2, 7999.00, 30);

INSERT INTO demo_20250120_products (product_name, category_id, price, stock)
VALUES ('小米电视', 3, 4999.00, 20);

INSERT INTO demo_20250120_products (product_name, category_id, price, stock)
VALUES ('戴尔笔记本', 2, 8999.00, 25);

INSERT INTO demo_20250120_products (product_name, category_id, price, stock)
VALUES ('索尼无线耳机', 4, 2999.00, 40);

INSERT INTO demo_20250120_products (product_name, category_id, price, stock)
VALUES ('佳能相机', 5, 24999.00, 10);

INSERT INTO demo_20250120_products (product_name, category_id, price, stock)
VALUES ('美的智能空调', 6, 3999.00, 35);

INSERT INTO demo_20250120_products (product_name, category_id, price, stock)
VALUES ('海尔滚筒洗衣机', 6, 2999.00, 45);

INSERT INTO demo_20250120_products (product_name, category_id, price, stock)
VALUES ('格力空气净化器', 6, 1999.00, 50);

INSERT INTO demo_20250120_products (product_name, category_id, price, stock)
VALUES ('联想ThinkPad', 2, 9999.00, 15);

INSERT INTO demo_20250120_products (product_name, category_id, price, stock)
VALUES ('三星手环', 1, 5999.00, 20);

INSERT INTO demo_20250120_products (product_name, category_id, price, stock)
VALUES ('大疆无人机', 4, 1999.00, 30);

INSERT INTO demo_20250120_products (product_name, category_id, price, stock)
VALUES ('尼康微波炉', 5, 16999.00, 10);

INSERT INTO demo_20250120_products (product_name, category_id, price, stock)
VALUES ('海信激光电视', 3, 9999.00, 15);

INSERT INTO demo_20250120_products (product_name, category_id, price, stock)
VALUES ('松下电饭煲', 6, 999.00, 60);

INSERT INTO demo_20250120_products (product_name, category_id, price, stock)
VALUES ('LG游戏电视', 3, 7999.00, 20);

INSERT INTO demo_20250120_products (product_name, category_id, price, stock)
VALUES ('飞利浦电动牙刷', 6, 399.00, 70);

INSERT INTO demo_20250120_products (product_name, category_id, price, stock)
VALUES ('小米蓝牙音箱', 4, 1499.00, 35);

INSERT INTO demo_20250120_products (product_name, category_id, price, stock)
VALUES ('索尼保温杯', 1, 3999.00, 10);

-- 创建 demo_20250120_categories 表
CREATE TABLE demo_20250120_categories (
    category_id SERIAL PRIMARY KEY,
    category_name VARCHAR(50) NOT NULL
);

COMMENT ON COLUMN demo_20250120_categories.category_id IS '类别ID';
COMMENT ON COLUMN demo_20250120_categories.category_name IS '类别名称';

INSERT INTO demo_20250120_categories (category_name)
VALUES ('智能手机');

INSERT INTO demo_20250120_categories (category_name)
VALUES ('笔记本电脑');

INSERT INTO demo_20250120_categories (category_name)
VALUES ('平板电视');

INSERT INTO demo_20250120_categories (category_name)
VALUES ('无线耳机');

INSERT INTO demo_20250120_categories (category_name)
VALUES ('数码相机');

INSERT INTO demo_20250120_categories (category_name)
VALUES ('家用空调');

INSERT INTO demo_20250120_categories (category_name)
VALUES ('滚筒洗衣机');

INSERT INTO demo_20250120_categories (category_name)
VALUES ('空气净化器');

INSERT INTO demo_20250120_categories (category_name)
VALUES ('游戏本');

INSERT INTO demo_20250120_categories (category_name)
VALUES ('平板电脑');

INSERT INTO demo_20250120_categories (category_name)
VALUES ('蓝牙音箱');

INSERT INTO demo_20250120_categories (category_name)
VALUES ('微单相机');

INSERT INTO demo_20250120_categories (category_name)
VALUES ('游戏电视');

INSERT INTO demo_20250120_categories (category_name)
VALUES ('电饭煲');

INSERT INTO demo_20250120_categories (category_name)
VALUES ('OLED电视');
INSERT INTO demo_20250120_categories (category_name)
VALUES ('电动牙刷');
INSERT INTO demo_20250120_categories (category_name)
VALUES ('游戏机');
INSERT INTO demo_20250120_categories (category_name)
VALUES ('智能手表');

'''
# 建立数据库连接
def create_conn():
    conn = None
    try:
        conn = psycopg2.connect(**db_params)
        print("Connection to PostgreSQL DB successful")
    except OperationalError as e:
        print(f"The error '{e}' occurred")
    return conn


# 执行SQL查询并返回结果
def execute_query(conn, query):
    cursor = conn.cursor()
    try:
        cursor.execute(query)
        conn.commit()  # 提交事务
        result = cursor.fetchall()
        column_names = [desc[0] for desc in cursor.description]
        return result, column_names
    except Exception as e:
        conn.rollback()  # 回滚事务
        print(f"Error executing query: {e}")
        cursor.close()
        raise ValueError("执行SQL报错：", e)


# 重连数据库
def reconnect_db():
    global db_connection
    if db_connection is not None and db_connection.closed:
        db_connection = create_conn()
    return db_connection


# sql 测试集
# [('SELECT', 'SELECT column1, column2 FROM table WHERE condition;'), ('INSERT', "INSERT INTO table (column1, column2) VALUES ('value1', 'value2');"), ('UPDATE', "UPDATE table SET column1 = 'new_value' WHERE condition;"), ('DELETE', 'DELETE FROM table WHERE condition;'), ('CREATE', 'CREATE TABLE new_table (column1 VARCHAR(255), column2 INT);'), ('DROP', 'DROP TABLE table;'), ('SELECT', 'WITH temp AS (SELECT * FROM table WHERE condition) SELECT * FROM temp;'), ('SELECT', 'SELECT t1.column1, t2.column2 FROM table1 t1 JOIN table2 t2 ON t1.id = t2.id;'), ('SELECT', 'WITH temp AS (SELECT * FROM table1) SELECT t1.column1, t2.column2 FROM temp t1 JOIN table2 t2 ON t1.id = t2.id;'), ('INSERT', 'WITH temp AS (SELECT * FROM source) INSERT INTO target SELECT * FROM temp;'), ('UPDATE', "WITH temp AS (SELECT * FROM source) UPDATE target SET column = 'value' WHERE id IN (SELECT id FROM temp);"), ('DELETE', 'WITH temp AS (SELECT * FROM source) DELETE FROM target WHERE id IN (SELECT id FROM temp);'), ('SELECT', 'SELECT * FROM (SELECT * FROM table WHERE condition) AS subquery;'), ('SELECT', 'WITH temp1 AS (SELECT * FROM table1), temp2 AS (SELECT * FROM table2) SELECT * FROM temp1 JOIN temp2 ON temp1.id = temp2.id;'), ('SELECT', 'SELECT t1.column1, t2.column2, t3.column3 FROM table1 t1 JOIN table2 t2 ON t1.id = t2.id JOIN table3 t3 ON t1.id = t3.id;'), ('SELECT', 'SELECT t1.column1, t2.column2 FROM table1 t1 JOIN (SELECT * FROM table2 WHERE condition) t2 ON t1.id = t2.id;'), ('SELECT', 'SELECT column1, COUNT(column2) FROM table GROUP BY column1 HAVING COUNT(column2) > 10;'), ('SELECT', 'SELECT column1, column2 FROM table ORDER BY column1 DESC LIMIT 10;'), ('SELECT', 'SELECT column1 FROM table1 UNION SELECT column1 FROM table2;'), ('SELECT', "SELECT column1, CASE WHEN column2 > 10 THEN 'High' ELSE 'Low' END AS status FROM table;"), ('SELECT', "WITH age_groups AS (SELECT doctor_uid, CASE WHEN age BETWEEN 0 AND 5 THEN '0-5' WHEN age BETWEEN 6 AND 10 THEN '6-10' WHEN age BETWEEN 11 AND 15 THEN '11-15' WHEN age BETWEEN 16 AND 20 THEN '16-20' WHEN age BETWEEN 21 AND 25 THEN '21-25' WHEN age BETWEEN 26 AND 30 THEN '26-30' WHEN age BETWEEN 31 AND 35 THEN '31-35' WHEN age BETWEEN 36 AND 40 THEN '36-40' WHEN age BETWEEN 41 AND 45 THEN '41-45' WHEN age BETWEEN 46 AND 50 THEN '46-50' WHEN age BETWEEN 51 AND 55 THEN '51-55' WHEN age BETWEEN 56 AND 60 THEN '56-60' WHEN age BETWEEN 61 AND 65 THEN '61-65' WHEN age BETWEEN 66 AND 70 THEN '66-70' WHEN age BETWEEN 71 AND 75 THEN '71-75' WHEN age BETWEEN 76 AND 80 THEN '76-80' WHEN age BETWEEN 81 AND 85 THEN '81-85' WHEN age BETWEEN 86 AND 90 THEN '86-90' WHEN age BETWEEN 91 AND 95 THEN '91-95' WHEN age BETWEEN 96 AND 100 THEN '96-100' ELSE '100+' END AS age_group FROM public.demo_table2) SELECT age_group, COUNT(*) AS people_count FROM age_groups GROUP BY age_group;"), ('SELECT', 'SELECT * FROM table'), ('INSERT', "INSERT INTO table (column) VALUES ('value')"), ('UPDATE', "UPDATE table SET column = 'value'"), ('DELETE', "DELETE FROM table WHERE column = 'value'"), ('CREATE', 'CREATE TABLE table (column VARCHAR(255))'), ('DROP', 'DROP TABLE table'), ('SELECT', "SELECT * FROM employees WHERE department = 'Sales';"), ('SELECT', 'SELECT e.employee_id, e.first_name, e.last_name, d.department_name, COUNT(*) as num_projects FROM employees e JOIN departments d ON e.department_id = d.department_id JOIN projects p ON e.employee_id = p.employee_id GROUP BY e.employee_id, d.department_name HAVING num_projects > 5;'), ('INSERT', "INSERT INTO employees (employee_id, first_name, last_name, email) VALUES (101, 'John', 'Doe', 'john.doe@example.com');"), ('INSERT', "INSERT INTO employees (employee_id, first_name, last_name, email, hire_date, job_id, salary, manager_id, department_id) VALUES (102, 'Jane', 'Smith', 'jane.smith@example.com', '2023-01-01', 'IT_PROG', 8000, 101, 90);"), ('UPDATE', 'UPDATE employees SET salary = salary * 1.1 WHERE department_id = 90;'), ('UPDATE', "UPDATE employees SET salary = CASE WHEN job_id IN ('IT_PROG', 'IT_MGR') THEN salary * 1.2 ELSE salary * 1.1 END WHERE department_id = 90;"), ('DELETE', 'DELETE FROM employees WHERE employee_id = 101;'), ('DELETE', 'DELETE FROM employees WHERE department_id NOT IN (SELECT department_id FROM departments WHERE location_id = 1700);'), ('CREATE', 'CREATE TABLE new_employees (employee_id INT PRIMARY KEY, first_name VARCHAR(50), last_name VARCHAR(50), email VARCHAR(100));'), ('CREATE', 'CREATE TABLE project_tasks (task_id INT PRIMARY KEY, project_id INT, task_description VARCHAR(255), due_date DATE, status VARCHAR(20), FOREIGN KEY (project_id) REFERENCES projects(project_id));'), ('DROP', 'DROP TABLE old_employees;'), ('DROP', 'DROP TABLE IF EXISTS old_employees;'), ('ALTER', 'ALTER TABLE employees ADD COLUMN middle_name VARCHAR(50);'), ('ALTER', 'ALTER TABLE employees MODIFY COLUMN email VARCHAR(150) UNIQUE;'), ('SELECT', 'WITH CityAverages AS (SELECT city, AVG(age) AS avg_age FROM public.demo_table2 GROUP BY city) ((SELECT city, avg_age FROM CityAverages ORDER BY avg_age DESC LIMIT 5) UNION ALL (SELECT city, avg_age FROM CityAverages ORDER BY avg_age ASC LIMIT 5))')]
def get_sql_type(sql):
    with_keyword = ""
    statements = sqlparse.parse(sql)
    for statement in statements:
        if statement.tokens[0].ttype in (
        sqlparse.tokens.Keyword, sqlparse.tokens.Keyword.DDL, sqlparse.tokens.Keyword.DML, sqlparse.tokens.Keyword.CTE):
            keyword = statement.tokens[0].value.upper()
            if keyword in ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'DROP', 'ALTER']:
                return keyword
            elif keyword in ['WITH']:
                for token in statement.tokens:
                    if token.ttype in (
                    sqlparse.tokens.Keyword, sqlparse.tokens.Keyword.DDL, sqlparse.tokens.Keyword.DML,
                    sqlparse.tokens.Keyword.CTE):
                        keyword = token.value.upper()
                        if keyword in ['SELECT']:
                            with_keyword = keyword
                        elif keyword in ['INSERT', 'UPDATE', 'DELETE', 'CREATE', 'DROP', 'ALTER']:
                            return keyword
                    elif isinstance(token, sqlparse.sql.Parenthesis):
                        inner_statements = [tok for tok in token.flatten() if tok.ttype in (
                        sqlparse.tokens.Keyword, sqlparse.tokens.Keyword.DDL, sqlparse.tokens.Keyword.DML,
                        sqlparse.tokens.Keyword.CTE)]
                        for inner_statement in inner_statements:
                            inner_keyword = inner_statement.value.upper()
                            if inner_keyword in ['SELECT', ]:
                                with_keyword = inner_keyword
                            elif inner_keyword in ['INSERT', 'UPDATE', 'DELETE', 'CREATE', 'DROP', 'ALTER']:
                                return inner_keyword

        elif isinstance(statement, sqlparse.sql.Parenthesis):
            # 如果是WITH语句，递归地解析内部的SQL语句
            inner_statements = [tok for tok in statement.flatten() if isinstance(tok, sqlparse.sql.Statement)]
            for inner_statement in inner_statements:
                inner_keyword = inner_statement.tokens[0].value.upper()
                if inner_keyword in ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'DROP', 'ALTER']:
                    return inner_keyword
    return with_keyword or 'UNKNOWN'


# 主函数，处理用户输入的SQL查询
def handle_sql_query(sql_query, format="df"):
    global db_connection
    if db_connection is None or db_connection.closed:
        db_connection = create_conn()

    try:
        result, columns = execute_query(db_connection, sql_query)
    except (Exception, psycopg2.DatabaseError) as error:
        print(f'执行SQL出错了:\n{sql_query}\n', error)
        db_connection.close()  # 显式关闭连接
        db_connection = create_conn()  # 重新建立连接
        try:
            result, columns = execute_query(db_connection, sql_query)
        except Exception as e:
            print('执行SQL出错了', e)
            if format == "df":
                return pd.DataFrame([str(e)], columns=['查询出错'])
            else:
                return [], []
    if format != "df":
        return columns, result
    print("sql查询结果：", result)
    df = pd.DataFrame(result, columns=columns)
    # Include the fixed text output
    print('查询结果：', df.head(5))

    return df


# 查询表字段名及数据类型：
def get_column_data_type(table_schema='public', table_name='demo_table2'):
    dbname = db_params['dbname']
    sql = f"""   SELECT
        column_name, data_type
        FROM information_schema.columns c
        WHERE table_catalog='{dbname}' and table_schema = '{table_schema}' and
        table_name = '{table_name}';
        """
    columns, results = handle_sql_query(sql, format='list')
    # print(columns,'\n', results)
    return {k: v for k, v in results}

# 查询表字段及字段注释：
def get_column_comment(table_schema='public', table_name='demo_table2'):
    sql = f"""SELECT
  col.attname AS column_name,
  pgd.description AS column_comment
FROM
  pg_attribute col
  LEFT JOIN pg_description pgd ON col.attrelid = pgd.objoid AND col.attnum = pgd.objsubid
WHERE
  col.attrelid = '{table_schema}{"." if table_schema else ""}{table_name}'::regclass
  AND col.attnum > 0
  AND NOT col.attisdropped
ORDER BY
  col.attnum;"""
    columns, results = handle_sql_query(sql, format='list')
    return {k: v for k, v in results}


# 查询各个字段，各个取值的频数
def get_column_values(table_schema='public', table_name='demo_table2', data_type_dict=None):
    sql_list = []
    text_columns = [col for col, data_type in data_type_dict.items() if data_type.lower().startswith("character") or data_type.lower().startswith("text")  or data_type.lower().startswith("json") ]
    other_columns  = [col for col, data_type in data_type_dict.items() if col not in text_columns]
    columns, results = [], []
    if text_columns:
        for col in text_columns:
            sql = f"""(SELECT '{col}' as field, {col}::text as value, COUNT(*) as num
         FROM {table_schema}{"." if table_schema else ""}{table_name}
         GROUP BY {col})"""
            sql_list.append(sql)

        sql = " UNION ".join(sql_list)
        print("sql=", sql)
        columns, results = handle_sql_query(sql, format='list')
    if other_columns:
        sql_list = []
        for col in other_columns:
            sql = f"""array_agg(DISTINCT {col}) as {col}"""
            sql_list.append(sql)

        sql = f"""SELECT {", ".join(sql_list)} FROM {table_schema}{"." if table_schema else ""}{table_name}"""
        print("sql=", sql)
        columns2, results2 = handle_sql_query(sql, format='list')
        print("columns, results: ", columns2, results2)
        if results2:
            for col, values in zip(columns2, results2[0]):
                for v in values:
                    results.append([col, v, 1])

    return {col: {k: v for c, k, v in results if c == col} for col in set([t[0] for t in results])}


def agents_chat(prompt):
    try:
        client = OpenAI(
            api_key=MODELSCOPE_SDK_TOKEN,  # ModelScope Token
            base_url="https://api-inference.modelscope.cn/v1"
        )
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        response = client.chat.completions.create(
            model="Qwen/Qwen2.5-72B-Instruct",
            messages=messages,
        )
        answer = response.choices[0].message.content
        print("返回结果：", answer)
        return answer

    except Exception as e:
        print("请求出错：", e)
        answer = str(e)
    return answer


def is_reset_command(command):
    reset_commands = [
        r'^.{0,3}清空对话.{0,3}$',
        r'^.{0,3}忘记上面所有.{0,3}$',
        r'^.{0,3}重置.{0,3}$',
        r'^.{0,3}清空消息.{0,3}$',
        r'^.{0,3}从头开始.{0,3}$',
        r'^.{0,3}忽略之前的所有内容.{0,3}$',
        r'^.{0,3}重新开始.{0,3}$',
        r'^.{0,3}忘记之前的?对话.{0,3}$',
        r'^.{0,3}清除历史记录.{0,3}$',
        r'^.{0,3}重置我们的?对话.{0,3}$',
        r'^.{0,3}忽略之前的?对话.{0,3}$',
        r'^.{0,3}清空聊天记录.{0,3}$',
        r'^.{0,3}((忘记)|(清除)|(忽略)|(重置)|(清空))[我们之前说的]*((所有话)|(历史消息)|(对话)|(消息)|(历史内容)|(记录)|(聊天)|(所有话)).{0,3}$',
        r'^.{0,3}忘记之前的聊天.{0,3}$',
        r'^.{0,3}((所有话)|(历史消息)|(对话)|(消息)|(历史内容)|(记录)|(聊天)|(所有话))[我们之前说的]*((忘记)|(清除)|(忽略)|(重置)|(清空)).{0,3}$',
        r'^.{0,3}可以重新开始.{0,3}$',
        r'^.{0,3}请把我们的对话重置到初始状态.{0,3}$',
        r'^.{0,3}请把我们的对话清空.{0,3}$',
        r'^.{0,3}请把我们的聊天历史清空.{0,3}$',
    ]
    for pattern in reset_commands:
        if re.match(pattern, command.strip()):
            return True
    return False


# 初始化数据库连接
db_connection = create_conn()


def chat(message, prompt, history, request: gr.Request):
    print("history", history)

    headers = request.headers
    host = request.client.host
    user_agent = request.headers["user-agent"]
    print("请求参数", {
        "ip": host,
        "user_agent": user_agent,
        "headers": headers,
    })

    if len(message) > 10:
        history_str = ''
    else:
        history_str = json.dumps([t['content'] for t in history[-20:] if t.get('role') == 'user'], ensure_ascii=False)
    question = prompt.replace("{{question}}", message).replace("{{history}}", history_str)
    question = question.replace("{{currentDate}}", datetime.now().strftime("%Y-%m-%d"))

    history.append({"role": "user", "content": message})

    # 判断是否要清空历史对话
    if is_reset_command(message):
        history.append({"role": "assistant", "content": "好的，消息已清空"})
        return "", gr.Textbox( value=""), gr.DataFrame(pd.DataFrame()), history[-2:]

    ret = agents_chat(question)
    df = pd.DataFrame()
    try:
        try:
            json_data = json.loads(ret)
        except Exception as e1:
            print(f"解析 {ret} , 出错：{e1}")
            # 定义正则表达式模式，不区分大小写
            pattern = re.compile(r'"(thought|sql)"\s*:\s*"([^"]*)"', re.IGNORECASE)
            # 查找所有匹配项
            matches = pattern.findall(ret)
            # 创建字典以存储结果
            json_data = {key: value for key, value in matches}
        text_response, sql_code = json_data['thought'], json_data['sql']
    except Exception as e:
        print("请求qwen出错：", e)
        text_response, sql_code = '请求出错', ''

    if sql_code:
        if get_sql_type(sql_code) != 'SELECT':
            df = pd.DataFrame(['SQL不是查询操作，暂不实际执行'], columns=['查询中止'])
        else:
            print('执行SQL：', sql_code)
            df = handle_sql_query(sql_code)

    history.append({"role": "assistant", "content": text_response})
    return "", gr.Textbox( value=sql_code, label='sql', scale=1), gr.DataFrame(df), history

def batch_test(prompt, excel_input, progress_demo=gr.Progress()):
    print(prompt)
    print(type(excel_input), excel_input)
    df = pd.read_excel(excel_input, dtype=str)
    datas = []

    gr.Info("开始测试", visible=False)
    progress_demo(0, desc="开始...") # 将进度条初始化为0，并显示描述

    for message, currentDate, keywords in progress_demo.tqdm(df[['问题', 'currentDate', 'keyword']].values, desc="完成"):
        currentDate = str(currentDate)
        question = prompt.replace("{{question}}", message).replace("{{history}}", "[]")
        question = question.replace("{{currentDate}}", currentDate)

        ret = agents_chat(question)
        df = pd.DataFrame()
        try:
            json_data = json.loads(ret)
            text_response, sql_code = json_data['thought'], json_data['sql']
        except Exception as e:
            print("请求qwen出错：", e)
            text_response, sql_code = '请求出错', ''

        if sql_code:
            if get_sql_type(sql_code) != 'SELECT':
                df = pd.DataFrame(['SQL不是查询操作，暂不实际执行'], columns=['查询中止'])
            else:
                print('执行SQL：', sql_code)
                df = handle_sql_query(sql_code)
        answer = json.dumps(df.astype(str).to_dict(), ensure_ascii=False)
        keywords = f"{keywords}".strip()
        retsult_label = "未知"
        if not answer or answer.strip() == '{}' or "查询出错" in json.loads(answer).keys():
            retsult_label = "错误"
        elif keywords and keywords.strip():
            for keywrod in keywords.strip().split('||'):
                keywrod = keywrod.strip()
                if keywrod:
                    if all(key in sql_code or key in answer for key in keywrod.split()):
                        retsult_label = "正确"
                        break
            else:
                retsult_label = "错误"
        else:
            retsult_label = "未知"

        datas.append([message, currentDate, ret, sql_code, answer, keywords, retsult_label])
    result_df = pd.DataFrame(datas, columns=['问题', "currentDate", "llm结果", "sql", '答案', 'keyword', '结果'])
    output_file = rf"./output/测试结果_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx"
    result_df.to_excel(output_file, index=False)

    df2 = pd.DataFrame([[len(datas), result_df[result_df['结果']=="正确"].shape[0] , result_df[result_df['结果']=="错误"].shape[0] , result_df[result_df['结果']=="未知"].shape[0] ,
          result_df[result_df['结果']=="正确"].shape[0]/len(datas), result_df[result_df['结果']=="错误"].shape[0]/len(datas)]], columns=['问题总数', "正确数", "错误数", "未知数", "正确率", "错误率"])
    return output_file, gr.DataFrame(df2)


def generator_prompt(table_schema='public', table_name_list=None, partition_time_col_dict=None, use_type=True):
    prompt_template = '''#Role：您是一位在 SQL 语言方面经验丰富的数据分析师. 
#Task：您的任务分为两步：
1. 您将获得一个用户提出的自然语言问题，请将其转换为 SQL 查询，以便通过对底层数据库执行 SQL 查询来返回相关数据。
2. 审查并修正生成的SQL，确保它满足以下标准：
   - 遵循特定数据库语法（本例中为{{DatabaseType}}）。
   - 只引用存在的表，以避免“undefined table“等错误。
   - 只引用存在的列，以避免如“列不存在”之类的错误。
   - 特别注意：在处理地理位置相关的查询时，确保省份和城市名称的准确性和与‘示例值’的一致性。具体规则如下：
      1) 如果用户输入的省份或城市名称与示例值中的后缀不一致（如“省”、“自治区”等），则需要在WHERE子句中调整名称以匹配示例值的格式。
      2) 如果用户输入的省份或城市名称包含了示例值中未出现的后缀，也应在WHERE子句中进行相应调整，以确保查询的准确性。
#Rules：
1.ALWAYS generate columns and values specified in the 'Schema', DO NOT hallucinate. 
2.ALWAYS be cautious, word in the 'Schema' does not mean it must appear in the SQL. 
3.ALWAYS specify date filter using '>','<','>=','<=' operator. 
4.DO NOT include date filter in the where clause if not explicitly expressed in the 'Question'. 
5.DO NOT calculate date range using functions. 
6.DO NOT miss the 'AGGREGATE' operator of metrics, always add it as needed. 
7.ALWAYS use 'with' statement if nested aggregation is needed. 
8.ALWAYS translate alias created by 'AS' command to english. 
9.若问题不明确，则根据最近的历史消息History中内容，判断需要查询的指标.

#Exemplars: 
{{exemplar}}
Schema: DatabaseType=[{{DatabaseType}}], 
{{TablesDimensions}} 
SideInfo：CurrentDate=[{{currentDate}}],
History: {{history}}
#Query: Question:{{question}}
You must answer strictly in the following JSON format: { 
"thought": (thought or remarks to tell users about the sql.; type: string), 
"sql": (sql to generate; type: string) 
}'''
    table_detail_dict = {}
    for table_name in table_name_list:
        partition_time_col_name = partition_time_col_dict.get(table_name, None)
        data_type_dict = get_column_data_type(table_schema, table_name)
        if partition_time_col_name is None:
            for col, data_type in data_type_dict.items():
                if data_type.lower().startswith("date"):
                    partition_time_col_name = col
                    break
        column_comments_dict = get_column_comment(table_schema=table_schema, table_name=table_name)
        column_values_dict = get_column_values(table_schema=table_schema, table_name=table_name, data_type_dict={k: v for k, v in data_type_dict.items() if k==partition_time_col_name}) if partition_time_col_name else {}

        strftime_dict = {
            '%Y-%m-%d':  'yyyy-MM-dd'
        }
        date_format = ''
        for pystr_format, _format in strftime_dict.items():
            for date_value, _ in sorted(column_values_dict.get(partition_time_col_name, {}).items(), key=lambda x: -x[1])[:10]:
                if date_value is None or date_value == "" or str(date_value).lower() == "null":
                    continue
                if (isinstance(date_value, str) and re.search(r'^\d{4}(0[1-9]|1[0-2])$', date_value.strip())):
                    date_format = "yyyyMM"
                    break
                try:
                    date_value.strftime(pystr_format)
                    date_format = _format
                except ValueError:
                    continue

        if partition_time_col_name:
            PartitionTime = f"{column_comments_dict.get(partition_time_col_name)} <{partition_time_col_name}> "
            if date_format:
                PartitionTime += f" FORMAT '{date_format}'"
        else:
            PartitionTime = ""

        if use_type:
            temp_data_type_dict = {col:data_type for col, data_type in data_type_dict.items() if data_type.lower().startswith("character") or data_type.lower().startswith("text") or data_type.lower().startswith("json")}
        else:
            temp_data_type_dict = data_type_dict
        column_values_dict = get_column_values(table_schema=table_schema, table_name=table_name, data_type_dict=temp_data_type_dict)
        dimensions = []
        for col, data_type in data_type_dict.items():
            values = [v for v, _ in sorted(column_values_dict.get(col, {}).items(), key=lambda x: -x[1]) if v is not None and v != "" and ((not isinstance(v, str)) or v.lower() != "null")]
            if len(values) <=5:
                values_list = [f"<{col}='{v}'>" if isinstance(v, str) else f"<{col}={v}>"  for v in values]
            elif not use_type:
                values_list = [f"<{col}='{v}'>" if isinstance(v, str) else f"<{col}={v}>"  for v in values[:5]]
            # elif all(re.search("^[^\u4E00-\u9FD5]*$", v) for v in values):
            #     values_list = []
            else:
                values_list = [f"<{col}='{v}'>" if isinstance(v, str) else f"<{col}={v}>"  for v in values[:5]]
            dimension = f"Dimensions=[{column_comments_dict.get(col, col)} <{col}>]"
            if use_type:
                dimension += f" DataType=[{data_type_dict[col].split()[0]}]"
            dimension += f" Values=[{" ".join(values_list)} ]"
            dimensions.append(dimension)
        table_detail_dict[table_name] = PartitionTime, dimensions
    TablesDimensions = ""
    for idx, (table_name, (PartitionTime, dimensions)) in enumerate(table_detail_dict.items(), 1):
        TablesDimensions += f"""Table{idx}=[{table_schema}{"." if table_schema else ""}{table_name}], 
PartitionTime=[{PartitionTime}], 
{',\n'.join(dimensions)};\n """
    prompt_template_replace_dict = {
        "{{DatabaseType}}": "postgresql",
        # "{{Table}}": f"{table_schema}{"." if table_schema else ""}{table_name}",
        # "{{PartitionTime}}": PartitionTime, # 数据日期字段
        "{{TablesDimensions}}": TablesDimensions
    }
    for keyword, content in prompt_template_replace_dict.items():
        prompt_template = prompt_template.replace(keyword, content)
    return prompt_template

def sentence_builder(dbType, table_schema, table_name, partition_time_col_name, use_type):
    global DATA_SOURCE_DICT
    print(f"输入参数, 数据库类型：{dbType}, 模型：{table_schema}, 表名：{table_name}, 数据日期字段：{partition_time_col_name}")
    if dbType not in ["PostgreSQL"]:
        return "该数据库类型不支持！"

    partition_time_col_dict = {x.split('||')[0]: x.split('||')[1] for x in partition_time_col_name}
    prompt_template = generator_prompt(table_schema=table_schema, table_name_list=table_name, partition_time_col_dict=partition_time_col_dict, use_type=use_type)
    print(prompt_template)
    if (isinstance(table_name, str)):
        key=(dbType, table_schema, table_name)
    else:
        key=(dbType, table_schema, ','.join(sorted(table_name)))
    DATA_SOURCE_DICT[key] = prompt_template
    return prompt_template

def update_dropdown():
    global DATA_SOURCE_DICT
    return gr.Dropdown(choices=['.'.join(dbt) for dbt in DATA_SOURCE_DICT.keys()])

def update_prompt(db_table):
    global DATA_SOURCE_DICT
    print("db_table", db_table)
    prompt = DATA_SOURCE_DICT.get(tuple(db_table.split('.')), '')
    return db_table, prompt

def update_table_schema(dbType):
    print("dbType", dbType)
    if dbType == "PostgreSQL":
        sql = "SELECT nspname AS schema_name FROM pg_catalog.pg_namespace"
        result, columns = handle_sql_query(sql, format='list')
        print("result, columns=", result, columns)
        # table_schema = [t[0] for t in columns]
        table_schema = [t[0] for t in columns]
    else:
        table_schema = []
    return dbType, gr.Dropdown(choices=table_schema)

def update_table_name(dbType, table_schema):
    if dbType == "PostgreSQL":
        sql = f"SELECT tablename FROM pg_catalog.pg_tables WHERE schemaname = '{table_schema}'"
        result, columns = handle_sql_query(sql, format='list')
        print("result, columns=", result, columns)
        # table_schema = [t[0] for t in columns]
        table_name = [t[0] for t in columns]
    else:
        table_name = []
    return table_schema, gr.Dropdown(choices=table_name)

def update_partition_time_col_name(dbType, table_schema, table_name):
    if dbType == "PostgreSQL":
        sql = f""" SELECT 
        c.table_name,
    c.column_name,
    c.data_type,
    d.description AS column_description
FROM 
    information_schema.columns c
LEFT JOIN 
    pg_catalog.pg_description d ON 
        d.objoid = c.table_name::regclass::oid 
        AND d.objsubid = c.ordinal_position
WHERE 
    c.table_schema = '{table_schema}'
    AND c.table_name in {tuple(table_name)} """
        result, columns = handle_sql_query(sql, format='list')
        print("result, columns=", result, columns)
        # table_schema = [t[0] for t in columns]
        columns = ["||".join(t) for t in columns]
    else:
        columns = []
    return gr.Dropdown(choices=columns)

DEFAULT_PROMPT = """#Role：您是一位在 SQL 语言方面经验丰富的数据分析师. 
#Task：您的任务分为两步：
1. 您将获得一个用户提出的自然语言问题，请将其转换为 SQL 查询，以便通过对底层数据库执行 SQL 查询来返回相关数据。
2. 审查并修正生成的SQL，确保它满足以下标准：
   - 遵循特定数据库语法（本例中为postgresql）。
   - 只引用存在的表，以避免“undefined table“等错误。
   - 只引用存在的列，以避免如“列不存在”之类的错误。
   - 特别注意：在处理地理位置相关的查询时，确保省份和城市名称的准确性和与‘示例值’的一致性。具体规则如下：
      1) 如果用户输入的省份或城市名称与示例值中的后缀不一致（如“省”、“自治区”等），则需要在WHERE子句中调整名称以匹配示例值的格式。
      2) 如果用户输入的省份或城市名称包含了示例值中未出现的后缀，也应在WHERE子句中进行相应调整，以确保查询的准确性。
#Rules：
1.ALWAYS generate columns and values specified in the 'Schema', DO NOT hallucinate. 
2.ALWAYS be cautious, word in the 'Schema' does not mean it must appear in the SQL. 
3.ALWAYS specify date filter using '>','<','>=','<=' operator. 
4.DO NOT include date filter in the where clause if not explicitly expressed in the 'Question'. 
5.DO NOT calculate date range using functions. 
6.DO NOT miss the 'AGGREGATE' operator of metrics, always add it as needed. 
7.ALWAYS use 'with' statement if nested aggregation is needed. 
8.ALWAYS translate alias created by 'AS' command to english. 
9.若问题不明确，则根据最近的历史消息History中内容，判断需要查询的指标.

#Exemplars: 
{{exemplar}}
Schema: DatabaseType=[postgresql], 
Table1=[public.demo_20250120_customers], 
PartitionTime=[], 
Dimensions=[客户ID <customer_id>] DataType=[integer] Values=[ ],
Dimensions=[姓名 <name>] DataType=[character] Values=[<name='高十六'> <name='孙七'> <name='张三'> <name='郑十'> <name='林十七'> ],
Dimensions=[年龄 <age>] DataType=[integer] Values=[ ],
Dimensions=[电话 <phone>] DataType=[character] Values=[<phone='13500000004'> <phone='15200000012'> <phone='15100000011'> <phone='15800000017'> <phone='15700000016'> ],
Dimensions=[省份 <province>] DataType=[character] Values=[<province='辽宁省'> <province='四川省'> <province='湖南省'> <province='广东省'> <province='山东省'> ],
Dimensions=[城市 <city>] DataType=[character] Values=[<city='渝中区'> <city='郑州市'> <city='浦东新区'> <city='合肥市'> <city='西安市'> ],
Dimensions=[地址 <address>] DataType=[text] Values=[<address='沈河区青年大街1号'> <address='陆家嘴环路1000号'> <address='东湖区八一大道333号'> <address='雁塔区小寨西路121号'> <address='长安区和平东路217号'> ];
 Table2=[public.demo_20250120_orders], 
PartitionTime=[下单日期 <order_date>  FORMAT 'yyyy-MM-dd'], 
Dimensions=[订单ID <order_id>] DataType=[integer] Values=[ ],
Dimensions=[客户ID <customer_id>] DataType=[integer] Values=[ ],
Dimensions=[下单日期 <order_date>] DataType=[timestamp] Values=[ ],
Dimensions=[订单总金额 <total_amount>] DataType=[numeric] Values=[ ];
 Table3=[public.demo_20250120_order_items], 
PartitionTime=[], 
Dimensions=[订单项ID <order_item_id>] DataType=[integer] Values=[ ],
Dimensions=[订单ID <order_id>] DataType=[integer] Values=[ ],
Dimensions=[产品ID <product_id>] DataType=[integer] Values=[ ],
Dimensions=[数量 <quantity>] DataType=[integer] Values=[ ],
Dimensions=[单价 <price>] DataType=[numeric] Values=[ ];
 Table4=[public.demo_20250120_products], 
PartitionTime=[], 
Dimensions=[产品ID <product_id>] DataType=[integer] Values=[ ],
Dimensions=[产品名称 <product_name>] DataType=[character] Values=[<product_name='松下电饭煲'> <product_name='三星手环'> <product_name='佳能相机'> <product_name='苹果电脑'> <product_name='联想ThinkPad'> ],
Dimensions=[类别ID <category_id>] DataType=[integer] Values=[ ],
Dimensions=[价格 <price>] DataType=[numeric] Values=[ ],
Dimensions=[库存量 <stock>] DataType=[integer] Values=[ ];
 Table5=[public.demo_20250120_categories], 
PartitionTime=[], 
Dimensions=[类别ID <category_id>] DataType=[integer] Values=[ ],
Dimensions=[类别名称 <category_name>] DataType=[character] Values=[<category_name='游戏机'> <category_name='数码相机'> <category_name='游戏电视'> <category_name='空气净化器'> <category_name='平板电脑'> ];
  
SideInfo：CurrentDate=[{{currentDate}}],
History: {{history}}
#Query: Question:{{question}}
You must answer strictly in the following JSON format: { 
"thought": (thought or remarks to tell users about the sql.; type: string), 
"sql": (sql to generate; type: string) 
}"""
with gr.Blocks() as demo:
    gr.Markdown("ChatBi")
    with gr.Tabs():
        with gr.TabItem("智能问答"):
            code = gr.Textbox(render=False, scale=1, label='sql')
            html = gr.DataFrame(render=False)
            with gr.Row():
                with gr.Column():
                    gr.Markdown("<center><h1>ChatBi</h1></center>")
                    with gr.Row():
                        db_table = gr.Dropdown([], label="数据源", info="若数据源列表为空，请先生成提示词", scale=10)
                        add_button = gr.Button("刷新数据源",size="sm", variant="primary")
                        add_button.click(update_dropdown, inputs=[], outputs=db_table)
                    prompt = gr.Textbox(value=DEFAULT_PROMPT, label="提示词", lines=9, max_lines=10)
                    db_table.change(update_prompt, inputs=[db_table], outputs=[db_table, prompt])
                    prompt.change(lambda x: x, inputs=prompt, outputs=prompt)
                    chatbot = gr.Chatbot(type="messages", label='问答对话框', height=400)
                    msg = gr.Textbox(label='请在这里输入你的问题', value='本月深圳的总订单金额？')
                    # 添加示例问题
                    examples = gr.Examples(
                        examples=[
                            '每个客户的总订单金额',
                             '每个产品的总销售数量',
                             '每个客户的订单详情，包括订单日期、产品名称和数量',
                             '每个类别的总销售额',
                             '每个客户的订单数量',
                             '每个产品的平均销售价格',
                             '每个客户的订单总金额和平均订单金额',
                             '每个类别的产品数量',
                             '每个客户的订单详情，包括订单日期、产品名称、数量和总金额',
                             '每个产品的总销售额',
                             '每个客户的订单详情，包括订单日期、产品名称、数量和总金额，并按总金额排序',
                             '每个类别的总销售数量',
                             '每个客户的订单详情，包括订单日期、产品名称、数量和总金额，并按客户ID排序',
                             '每个产品的平均销售数量',
                             '每个客户的订单详情，包括订单日期、产品名称、数量和总金额，并按订单日期排序',
                             '每个类别的总销售额和总销售数量',
                             '每个客户的订单详情，包括订单日期、产品名称、数量和总金额，并按产品名称排序',
                             '每个产品的总销售额和总销售数量',
                             '每个客户的订单详情，包括订单日期、产品名称、数量和总金额，并按总金额降序排序',
                             '每个类别的总销售额和总销售数量，并按总销售额降序排序'
                        ],
                        inputs=msg,
                        fn=lambda x: x,  # 简单的函数，用于将示例直接传递给输入框
                        cache_examples=False
                    )
                with gr.Column():
                    gr.Markdown("<center><h1>查询的SQL</h1></center>")
                    code.render()
                    gr.Markdown("<center><h1>查询的结果</h1></center>")
                    html.render()

            msg.submit(chat, [msg, prompt, chatbot], [msg, code, html, chatbot])
        with gr.TabItem("批量测试"):
            with gr.Row():
                # prompt = gr.Textbox(value="prompt2", label="提示词", lines=9, max_lines=20, scale=1)
                with gr.Column():
                    with gr.Row():
                        db_table = gr.Dropdown([], label="数据源", info="若数据源列表为空，请先生成提示词", scale=10)
                        add_button = gr.Button("刷新数据源", size="sm", variant="primary")
                        add_button.click(update_dropdown, inputs=[], outputs=db_table)
                    prompt = gr.Textbox(value="", label="提示词", lines=9, max_lines=10)
                    db_table.change(update_prompt, inputs=[db_table], outputs=[db_table, prompt])
                    prompt.change(lambda x: x, inputs=prompt, outputs=prompt)
                excel_input = gr.File(label="上传测试文件(.xlsx)")
            excel_button = gr.Button("开始测试", variant="primary")
            excel_output = gr.File(label="下载测试结果")
            ret_df = gr.DataFrame(label="测试结果统计")
            excel_button.click(batch_test, inputs=[prompt, excel_input], outputs=[excel_output, ret_df])
        with gr.TabItem("生成提示词(prompt)"):
            with gr.Row():
                with gr.Column():
                    dbType = gr.Dropdown(
                        ["PostgreSQL"], value='--', label="数据库类型", info="请选择数据量类型!"
                    )
                    # table_schema = gr.Textbox(value="public", label="模式")
                    table_schema = gr.Dropdown([], label="模式", info="", allow_custom_value=True)
                    dbType.change(update_table_schema, inputs=[dbType], outputs=[dbType, table_schema])
                    table_name = gr.Dropdown([], label="表名", multiselect=True)
                    table_schema.change(update_table_name, inputs=[dbType, table_schema], outputs=[table_schema, table_name])
                    table_name.change(lambda x: x, inputs=table_name, outputs=table_name)
                    # partition_time_col_name = gr.Textbox(value="treat_date", label="数据日期字段")

                    partition_time_col_name = gr.Dropdown([], label="数据日期字段", allow_custom_value=True, multiselect=True)
                    table_name.change(update_partition_time_col_name, inputs=[dbType, table_schema, table_name], outputs=[ partition_time_col_name])
                    partition_time_col_name.change(lambda x:x, inputs=partition_time_col_name, outputs=partition_time_col_name)

                    use_type = gr.Checkbox(label="添加字段类型", info="是否添加每个字段的数据类型?", value=True)

                prompt_template = gr.Textbox(label="prompt")
            submit_btn = gr.Button("一键生成prompt", variant="primary")
            submit_btn.click(sentence_builder, inputs=[dbType, table_schema, table_name, partition_time_col_name, use_type], outputs=[prompt_template])


# 启动Gradio应用
demo.launch(server_name='0.0.0.0', server_port=7860, share=True)


def main():
    pass


if __name__ == "__main__":
    main()
