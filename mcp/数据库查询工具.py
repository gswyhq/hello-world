#!/usr/bin/env python
# coding=utf-8


# mysql_streamable-http.py
from mcp.server.fastmcp import FastMCP, Context
from mysql.connector import connect, Error
from dotenv import load_dotenv
import os
import logging
import sqlparse

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# 创建MCP服务器
mcp = FastMCP("MySQL Query Tool", port=8005)


def get_db_config():
    """从环境变量获取数据库配置"""
    # 加载.env文件, .env文件内容如下
    # MYSQL_HOST=localhost
    # MYSQL_PORT=3306
    # MYSQL_USER=root
    # ...
    load_dotenv()

    config = {
        "host": os.getenv("MYSQL_HOST", "localhost"),
        "port": int(os.getenv("MYSQL_PORT", "3306")),
        "user": os.getenv("MYSQL_USER"),
        "password": os.getenv("MYSQL_PASSWORD"),
        "database": os.getenv("MYSQL_DATABASE"),
    }
    logger.info(f"config={config}")

    if not all([config["user"], config["password"], config["database"]]):
        raise ValueError("缺少必需的数据库配置")

    return config


# sql 测试集
def get_sql_type(sql):
    with_keyword = ""
    statements = sqlparse.parse(sql)
    for statement in statements:
        if statement.tokens[0].ttype in (
        sqlparse.tokens.Keyword, sqlparse.tokens.Keyword.DDL, sqlparse.tokens.Keyword.DML, sqlparse.tokens.Keyword.CTE):
            keyword = statement.tokens[0].value.upper()
            if keyword in ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'DROP', 'ALTER', 'DESCRIBE', 'SHOW']:
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

@mcp.tool()
def execute_sql(query: str) -> str:
    """执行SQL查询语句

    参数:
        query (str): 要执行的SQL语句，支持多条语句以分号分隔

    返回:
        str: 查询结果，格式化为可读的文本
    """
    config = get_db_config()
    logger.info(f"执行SQL查询: {query}")

    try:
        with connect(**config) as conn:
            with conn.cursor() as cursor:
                statements = [stmt.strip() for stmt in query.split(";") if stmt.strip()]
                results = []

                for statement in statements:
                    try:
                        if get_sql_type(statement) not in ('DESCRIBE', 'SHOW', 'SELECT'):
                            logger.info(f'SQL不是查询操作，暂不实际执行: {statement}')
                            results.append("SQL不是查询操作，暂不实际执行")
                            continue
                        cursor.execute(statement)

                        # 检查语句是否返回了结果集
                        if cursor.description:
                            columns = [desc[0] for desc in cursor.description]
                            rows = cursor.fetchall()

                            # 格式化输出
                            result = [" | ".join(columns)]
                            result.append("-" * len(result[0]))

                            for row in rows:
                                formatted_row = [str(val) if val is not None else "NULL" for val in row]
                                result.append(" | ".join(formatted_row))

                            results.append("\n".join(result))
                        else:
                            conn.commit()  # 提交非查询语句
                            results.append(f"查询执行成功。影响行数: {cursor.rowcount}")

                    except Error as stmt_error:
                        results.append(f"执行语句 '{statement}' 出错: {str(stmt_error)}")

                return "\n\n".join(results)

    except Error as e:
        error_msg = f"执行SQL '{query}' 时出错: {e}"
        logger.error(error_msg)
        return error_msg


@mcp.tool()
def get_table_structure(table_name: str) -> str:
    """获取指定表的结构信息

    参数:
        table_name (str): 表名

    返回:
        str: 表结构信息，包含字段名、类型、是否为NULL和默认值
    """
    query = f"DESCRIBE {table_name};"
    return execute_sql(query)


@mcp.tool()
def list_tables() -> str:
    """列出数据库中的所有表"""
    query = "SHOW TABLES;"
    return execute_sql(query)


@mcp.resource("db://tables")
def get_tables_resource() -> list:
    """获取数据库中的所有表名作为资源"""
    config = get_db_config()

    try:
        with connect(**config) as conn:
            with conn.cursor() as cursor:
                cursor.execute("SHOW TABLES;")
                return [row[0] for row in cursor.fetchall()]
    except Error as e:
        logger.error(f"获取表列表时出错: {e}")
        return []


@mcp.resource("db://schema/{table}")
def get_table_schema(table: str) -> dict:
    """获取指定表的模式定义作为资源"""
    config = get_db_config()

    try:
        with connect(**config) as conn:
            with conn.cursor() as cursor:
                # 获取表结构
                cursor.execute(f"DESCRIBE {table};")
                columns = cursor.fetchall()

                schema = {
                    "table": table,
                    "columns": []
                }

                for col in columns:
                    schema["columns"].append({
                        "name": col[0],
                        "type": col[1],
                        "nullable": col[2] == "YES",
                        "key": col[3],
                        "default": col[4],
                        "extra": col[5]
                    })

                return schema
    except Error as e:
        logger.error(f"获取表 {table} 的模式时出错: {e}")
        return {"error": str(e)}

def main():
    logger.info("启动MySQL查询工具")
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()

'''
# 客户端请求测试：
from mcp.client.streamable_http import streamablehttp_client
from mcp import ClientSession
# Connect to a streamable HTTP server
async def test_tool():
    # Connect to a streamable HTTP server
    async with streamablehttp_client("http://192.168.3.34:8005/mcp") as (
        read_stream,
        write_stream,
        _,
    ):
        # Create a session using the client streams
        async with ClientSession(read_stream, write_stream) as session:
            # Initialize the connection
            await session.initialize()
            # Call a tool
            tool_result = await session.call_tool("execute_sql", {'query': 'SELECT * FROM db1.table2 limit 5'})
            return tool_result
            
res = asyncio.run(test_tool())
[content.text for content in res.content]
'''