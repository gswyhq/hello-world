#!/usr/bin/env python
# coding=utf-8

# simple.py
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Simple Server", host='0.0.0.0', port=8042)


@mcp.tool()
def add(a: int, b: int) -> int:
    """这是个加法器，用于计算两个数字和"""
    print(f"请求计算：{a}, {b}")
    return a + b

@mcp.tool()
def get_weather(location:str) -> str:
    """查询某地的天气情况
    参数：location：需要查询天气的城市名
    """
    return f"`{location}`天气恶劣，近几天为台风天气"

@mcp.resource("greeting://{name}")
def get_greeting(name: str) -> str:
    """获取个性化问候"""
    return f"Hello, {name}!"



def main():
    # 初始化并运行 server
    print("mcp服务器启动")
    mcp.run(transport="sse")


if __name__ == "__main__":
    main()

