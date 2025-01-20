#!/usr/bin/env python
# coding=utf-8


import gradio as gr
import psycopg2
from psycopg2 import OperationalError
import pandas as pd

# 数据库连接参数
db_params = {
    'dbname': 'db123',
    'user': 'devdata',
    'password': '7854@2021',
    'host': '30.188.19.44',
    'port': '15432'
}

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

# 主函数，处理用户输入的SQL查询
def handle_sql_query(sql_query):

    global db_connection
    if db_connection is None or db_connection.closed:
        db_connection = create_conn()

    try:
        result, columns = execute_query(db_connection, sql_query)
    except (Exception, psycopg2.DatabaseError) as error:
        print('执行SQL出错了', error)
        db_connection.close()  # 显式关闭连接
        db_connection = create_conn()  # 重新建立连接
        try:
            result, columns = execute_query(db_connection, sql_query)
        except Exception as e:
            print('执行SQL出错了', e)
            return pd.DataFrame([str(e)])

    df = pd.DataFrame(result, columns=columns)

    fixed_text = f"根据你的SQL：{sql_query}，查询结果如下"
    print('查询结果：', df.head(5))

    return fixed_text, df


# 初始化数据库连接
db_connection = create_conn()

# Gradio界面
iface = gr.Interface(
    fn=handle_sql_query,
    inputs=gr.Textbox(value="SELECT * FROM public.demo_name limit 10", label='查询SQL'),  # 设置默认SQL查询语句
    outputs=[
        "text",
        "dataframe",  # 查询结果
    ],
    title="PostgreSQL Query Executor",
    description="输入要对 PostgreSQL 数据库执行的 SQL 进行查询."
)

# 启动Gradio应用
iface.launch(server_name='0.0.0.0', server_port=7860, share=True)

def main():
    pass


if __name__ == "__main__":
    main()
