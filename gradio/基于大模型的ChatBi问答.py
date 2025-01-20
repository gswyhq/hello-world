#!/usr/bin/env python
# coding=utf-8

import os
import re, json
import requests
import sqlparse
from datetime import datetime

USERNAME = os.getenv("USERNAME")

prompt = """#Role：您是一位在 SQL 语言方面经验丰富的数据分析师。
#Task：您将获得一个用户提出的自然语言问题，请将其转换为 SQL 查询，以便通过对底层数据库执行 SQL 查询来返回相关数据。
#Rules：
1.ALWAYS generate columns and values specified in the `Schema`, DO NOT hallucinate.
2.ALWAYS be cautious, word in the `Schema` does not mean it must appear in the SQL.
3.ALWAYS specify date filter using `>`,`<`,`>=`,`<=` operator.
4.DO NOT include date filter in the where clause if not explicitly expressed in the `Question`.
5.DO NOT calculate date range using functions.
6.DO NOT miss the AGGREGATE operator of metrics, always add it as needed.
7.ALWAYS use `with` statement if nested aggregation is needed.
8.ALWAYS enclose alias created by `AS` command in underscores.
9.ALWAYS translate alias created by `AS` command to the same language as the `#Question`.
10.若问句中没有提到指标，请根据历史记录判断需要查询的指标
#Exemplars:
Schema:
DatabaseType=[postgresql],
Table=[public.table3],
PartitionTime=[日期 <treat_date> FORMAT 'yyyy-MM-dd'],
ForeignKey=[客户ID <doctor_uid>],
Dimensions=[客户ID <doctor_uid>  AGGREGATE 'COUNT DISTINCT' ] Values=[ ],
Dimensions=[ 公司 <insur_company>] Values=[<insur_company='百度'> <insur_company='腾讯'> <insur_company='阿里'>  ],
Dimensions=[客户类型 <is_new_customer_cn>], Values=[<is_new_customer_cn='新'> <is_new_customer_cn='存量'> ],
Dimensions=[ 购买医院 <treat_hospital>], Values=[ ],
Dimensions=[购买日期 <treat_date> FORMAT 'yyyy-MM-dd'], Values=[ ],
Dimensions=[ 购买商品 <treat_result_cd10>], Values=[ ],
Dimensions=[是否成功 <is_guide_success_cn>], Values=[<is_guide_success_cn='是'> <is_guide_success_cn='否'> ],
Dimensions=[年龄 <age>] Values=[],
Dimensions=[性别 <gender>] Values=[<gender='男'> <gender='女'> ],
Dimensions=[省份 <province>], Values=[],
Dimensions=[城市 <city>], Values=[温州市,南京市...],
Dimensions=[购买次数 <treat_times>  AGGREGATE 'SUM' ] Values=[ ],
SideInfo：CurrentDate=[{{currentDate}}],
History: {{history}}
#Query: Question:{{question}}
Schema:DatabaseType=[postgresql], Table=[public.table3], PartitionTimeField=[treat_date FORMAT 'yyyy-MM-dd'], SideInfo：CurrentDate=[2024-12-18]
You must answer strictly in the following JSON format: {
"thought": (thought or remarks to tell users about the sql.; type: string),
"sql": (sql to generate; type: string)
}"""

# 广州市2024年购买量是多少，
# 百度的客户数是多少

# 实现一个选择读取展示excel表的页面，选择下拉列表，展示表格内容
import gradio as gr
import pandas as pd
import time  # 导入time模块，虽然在这个例子中未直接使用，但可用于其他功能如延迟
import psycopg2
from psycopg2 import OperationalError

# 数据库连接参数
db_params = {
    'dbname': 'bord',
    'user': 'boddata',
    'password': 'Pan@2021',
    'host': '30.112.80.5',
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


# sql 测试集
# [('SELECT', 'SELECT column1, column2 FROM table WHERE condition;'), ('INSERT', "INSERT INTO table (column1, column2) VALUES ('value1', 'value2');"), ('UPDATE', "UPDATE table SET column1 = 'new_value' WHERE condition;"), ('DELETE', 'DELETE FROM table WHERE condition;'), ('CREATE', 'CREATE TABLE new_table (column1 VARCHAR(255), column2 INT);'), ('DROP', 'DROP TABLE table;'), ('SELECT', 'WITH temp AS (SELECT * FROM table WHERE condition) SELECT * FROM temp;'), ('SELECT', 'SELECT t1.column1, t2.column2 FROM table1 t1 JOIN table2 t2 ON t1.id = t2.id;'), ('SELECT', 'WITH temp AS (SELECT * FROM table1) SELECT t1.column1, t2.column2 FROM temp t1 JOIN table2 t2 ON t1.id = t2.id;'), ('INSERT', 'WITH temp AS (SELECT * FROM source) INSERT INTO target SELECT * FROM temp;'), ('UPDATE', "WITH temp AS (SELECT * FROM source) UPDATE target SET column = 'value' WHERE id IN (SELECT id FROM temp);"), ('DELETE', 'WITH temp AS (SELECT * FROM source) DELETE FROM target WHERE id IN (SELECT id FROM temp);'), ('SELECT', 'SELECT * FROM (SELECT * FROM table WHERE condition) AS subquery;'), ('SELECT', 'WITH temp1 AS (SELECT * FROM table1), temp2 AS (SELECT * FROM table2) SELECT * FROM temp1 JOIN temp2 ON temp1.id = temp2.id;'), ('SELECT', 'SELECT t1.column1, t2.column2, t3.column3 FROM table1 t1 JOIN table2 t2 ON t1.id = t2.id JOIN table3 t3 ON t1.id = t3.id;'), ('SELECT', 'SELECT t1.column1, t2.column2 FROM table1 t1 JOIN (SELECT * FROM table2 WHERE condition) t2 ON t1.id = t2.id;'), ('SELECT', 'SELECT column1, COUNT(column2) FROM table GROUP BY column1 HAVING COUNT(column2) > 10;'), ('SELECT', 'SELECT column1, column2 FROM table ORDER BY column1 DESC LIMIT 10;'), ('SELECT', 'SELECT column1 FROM table1 UNION SELECT column1 FROM table2;'), ('SELECT', "SELECT column1, CASE WHEN column2 > 10 THEN 'High' ELSE 'Low' END AS status FROM table;"), ('SELECT', "WITH age_groups AS (SELECT doctor_uid, CASE WHEN age BETWEEN 0 AND 5 THEN '0-5' WHEN age BETWEEN 6 AND 10 THEN '6-10' WHEN age BETWEEN 11 AND 15 THEN '11-15' WHEN age BETWEEN 16 AND 20 THEN '16-20' WHEN age BETWEEN 21 AND 25 THEN '21-25' WHEN age BETWEEN 26 AND 30 THEN '26-30' WHEN age BETWEEN 31 AND 35 THEN '31-35' WHEN age BETWEEN 36 AND 40 THEN '36-40' WHEN age BETWEEN 41 AND 45 THEN '41-45' WHEN age BETWEEN 46 AND 50 THEN '46-50' WHEN age BETWEEN 51 AND 55 THEN '51-55' WHEN age BETWEEN 56 AND 60 THEN '56-60' WHEN age BETWEEN 61 AND 65 THEN '61-65' WHEN age BETWEEN 66 AND 70 THEN '66-70' WHEN age BETWEEN 71 AND 75 THEN '71-75' WHEN age BETWEEN 76 AND 80 THEN '76-80' WHEN age BETWEEN 81 AND 85 THEN '81-85' WHEN age BETWEEN 86 AND 90 THEN '86-90' WHEN age BETWEEN 91 AND 95 THEN '91-95' WHEN age BETWEEN 96 AND 100 THEN '96-100' ELSE '100+' END AS age_group FROM public.demo_table2) SELECT age_group, COUNT(*) AS people_count FROM age_groups GROUP BY age_group;"), ('SELECT', 'SELECT * FROM table'), ('INSERT', "INSERT INTO table (column) VALUES ('value')"), ('UPDATE', "UPDATE table SET column = 'value'"), ('DELETE', "DELETE FROM table WHERE column = 'value'"), ('CREATE', 'CREATE TABLE table (column VARCHAR(255))'), ('DROP', 'DROP TABLE table'), ('SELECT', "SELECT * FROM employees WHERE department = 'Sales';"), ('SELECT', 'SELECT e.employee_id, e.first_name, e.last_name, d.department_name, COUNT(*) as num_projects FROM employees e JOIN departments d ON e.department_id = d.department_id JOIN projects p ON e.employee_id = p.employee_id GROUP BY e.employee_id, d.department_name HAVING num_projects > 5;'), ('INSERT', "INSERT INTO employees (employee_id, first_name, last_name, email) VALUES (101, 'John', 'Doe', 'john.doe@example.com');"), ('INSERT', "INSERT INTO employees (employee_id, first_name, last_name, email, hire_date, job_id, salary, manager_id, department_id) VALUES (102, 'Jane', 'Smith', 'jane.smith@example.com', '2023-01-01', 'IT_PROG', 8000, 101, 90);"), ('UPDATE', 'UPDATE employees SET salary = salary * 1.1 WHERE department_id = 90;'), ('UPDATE', "UPDATE employees SET salary = CASE WHEN job_id IN ('IT_PROG', 'IT_MGR') THEN salary * 1.2 ELSE salary * 1.1 END WHERE department_id = 90;"), ('DELETE', 'DELETE FROM employees WHERE employee_id = 101;'), ('DELETE', 'DELETE FROM employees WHERE department_id NOT IN (SELECT department_id FROM departments WHERE location_id = 1700);'), ('CREATE', 'CREATE TABLE new_employees (employee_id INT PRIMARY KEY, first_name VARCHAR(50), last_name VARCHAR(50), email VARCHAR(100));'), ('CREATE', 'CREATE TABLE project_tasks (task_id INT PRIMARY KEY, project_id INT, task_description VARCHAR(255), due_date DATE, status VARCHAR(20), FOREIGN KEY (project_id) REFERENCES projects(project_id));'), ('DROP', 'DROP TABLE old_employees;'), ('DROP', 'DROP TABLE IF EXISTS old_employees;'), ('ALTER', 'ALTER TABLE employees ADD COLUMN middle_name VARCHAR(50);'), ('ALTER', 'ALTER TABLE employees MODIFY COLUMN email VARCHAR(150) UNIQUE;'), ('SELECT', 'WITH CityAverages AS (SELECT city, AVG(age) AS avg_age FROM public.demo_table2 GROUP BY city) ((SELECT city, avg_age FROM CityAverages ORDER BY avg_age DESC LIMIT 5) UNION ALL (SELECT city, avg_age FROM CityAverages ORDER BY avg_age ASC LIMIT 5))')]
def get_sql_type(sql):
    with_keyword = ""
    statements = sqlparse.parse(sql)
    for statement in statements:
        if statement.tokens[0].ttype in (sqlparse.tokens.Keyword, sqlparse.tokens.Keyword.DDL, sqlparse.tokens.Keyword.DML, sqlparse.tokens.Keyword.CTE):
            keyword = statement.tokens[0].value.upper()
            if keyword in ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'DROP', 'ALTER']:
                return keyword
            elif keyword in ['WITH']:
                for token in statement.tokens:
                    if token.ttype in (sqlparse.tokens.Keyword, sqlparse.tokens.Keyword.DDL, sqlparse.tokens.Keyword.DML, sqlparse.tokens.Keyword.CTE):
                        keyword = token.value.upper()
                        if keyword in ['SELECT']:
                            with_keyword = keyword
                        elif keyword in ['INSERT', 'UPDATE', 'DELETE', 'CREATE', 'DROP', 'ALTER']:
                            return keyword
                    elif isinstance(token, sqlparse.sql.Parenthesis):
                        inner_statements = [tok for tok in token.flatten() if tok.ttype in (sqlparse.tokens.Keyword, sqlparse.tokens.Keyword.DDL, sqlparse.tokens.Keyword.DML, sqlparse.tokens.Keyword.CTE)]
                        for inner_statement in inner_statements:
                            inner_keyword = inner_statement.value.upper()
                            if inner_keyword in ['SELECT',]:
                                with_keyword = inner_keyword
                            elif inner_keyword in [ 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'DROP', 'ALTER']:
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
    # Include the fixed text output
    print('查询结果：', df.head(5))

    return df


# 创建一个会话
# 调用/conversations的POST接口，接口返回新创建的conversation_id。
# API接入示例 - 创建conversation
# curl -XPOST  'http://chat-gpt/api/v1/conversations' -H 'Content-Type:application/json' -d '{"bot_id":"503"}'
def get_conversation_id(url='http://chat-gpt/api/v1/conversations', bot_id="503"):
    try:
        r = requests.post(url, headers={"Content-Type": "application/json"}, json={"bot_id": bot_id})
        ret = r.json()
        if ret.get('code') == 'success':
            return ret['data']['conversation_id']
    except Exception as e:
        print("请求会话id出错：", e)
    return ""


# API接入说明 - 发送消息
# 调用/conversations/{conversation_id}/messages的POST接口，使用上一步的conversation_id
# API接入示例 - 发送消息
# curl -XPOST 'http://chat-gpt/api/v1/conversations/728b73f6-4814-4d27-9200-44866fbe054b/messages' -H 'Content-Type:application/json' -d '{"message":{"content":"你好"}}'

def agents_chat(question, conversation_id, url='http://chat-gpt/api/v1/conversations'):
    try:
        url = f'{url}/{conversation_id}/messages'
        headers = {"Content-Type": "application/json"}
        query = {"message": {"content": question}}
        print(f"请求命令：\n\ncurl -XPOST {url} \\")
        for k, v in headers.items():
            print(f"-H '{k}: {v}' \\")
        print(f"-d '{json.dumps(query, ensure_ascii=False)}'")
        r = requests.post(url, headers=headers, json=query)
        ret = r.json()
        if ret['code'] == 'success':
            answer = ret['data'][0]['content']
        else:
            answer = r.text
    except Exception as e:
        print("请求会话id出错：", e)
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
conversation_id = get_conversation_id(url='http://chatgpt001/api/v1/conversations', bot_id="101")


def chat(message, history, request: gr.Request):
    print("history", history)

    headers = request.headers
    host = request.client.host
    user_agent = request.headers["user-agent"]
    print("请求参数", {
        "ip": host,
        "user_agent": user_agent,
        "headers": headers,
    })

    global conversation_id
    if len(message) > 10:
        history_str = ''
    else:
        history_str = json.dumps([t['content'] for t in history[-20:] if t.get('role') == 'user'], ensure_ascii=False)
    question = prompt2.replace("{{question}}", message).replace("{{history}}", history_str)
    question = question.replace("{{currentDate}}", datetime.now().strftime("%Y-%m-%d"))

    history.append({"role": "user", "content": message})

    # 判断是否要清空历史对话
    if is_reset_command(message):
        conversation_id = get_conversation_id(url='http://openai.com.cn/api/v1/conversations',
                                              bot_id="101")
        history.append({"role": "assistant", "content": "好的，消息已清空"})
        return "", gr.Code(language="sql", value=""), gr.HTML(""), history[-2:]

    ret = agents_chat(question, conversation_id, url='http://openai.com.cn/api/v1/conversations')
    df = pd.DataFrame()
    try:
        json_data = json.loads(ret)
        text_response, sql_code = json_data['thought'], json_data['sql']
    except Exception as e:
        print("请求大模型出错：", e)
        text_response, sql_code = '请求出错', ''

    if sql_code:
        if get_sql_type(sql_code) != 'SELECT':
            df = pd.DataFrame(['SQL不是查询操作，暂不实际执行'])
        else:
            print('执行SQL：', sql_code)
            df = handle_sql_query(sql_code)

    history.append({"role": "assistant", "content": text_response})
    return "", gr.Code(language="sql", value=sql_code), gr.HTML(df.to_html(index=False)), history


with gr.Blocks() as demo:
    code = gr.Code(render=False)
    html = gr.HTML(render=False)
    with gr.Row():
        with gr.Column():
            gr.Markdown("<center><h1>ChatBi</h1></center>")
            chatbot = gr.Chatbot(type="messages", label='问答对话框', height=700)
            msg = gr.Textbox(label='请在这里输入你的问题', value='今天天气怎么样？')
        with gr.Column():
            gr.Markdown("<center><h1>查询的SQL</h1></center>")
            code.render()
            gr.Markdown("<center><h1>查询的结果</h1></center>")
            html.render()
    msg.submit(chat, [msg, chatbot], [msg, code, html, chatbot])

# 启动Gradio应用
demo.launch(server_name='0.0.0.0', server_port=7860, share=True, auth=('admin', "123456"))


def main():
    pass


if __name__ == "__main__":
    main()
