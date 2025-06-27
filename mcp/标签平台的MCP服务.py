#!/usr/bin/env python
# coding=utf-8
# mcp_server_tag_platform/main.py

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from mcp.server.fastmcp import FastMCP, Context
from contextlib import asynccontextmanager
import os
import httpx
import json
from typing import AsyncGenerator, Dict, Any

mcp = FastMCP("tag-platform-mcp", port=8003)

# 配置项
DEFAULT_API_KEY = "b5b3db94f086bb2a54154ba4244402be"
TAG_API_KEY = os.getenv("TAG_API_KEY", DEFAULT_API_KEY)
TAG_API_ENDPOINT = "http://localhost:8001/v1"

def wrap_sse_response(data: dict) -> str:
    """包装符合调用方要求的响应结构"""
    return json.dumps(data, ensure_ascii=False)


def _format_tag_data(raw_data: dict) -> dict:
    """标准化标签数据结构"""
    return {
        "user_id": raw_data["user_id"],
        "base_tags": [{"name": t["name"], "weight": t["score"]} for t in raw_data["basic_tags"]],
        "behavior_tags": [{"name": t["tag_name"], "last_active": t["update_time"]} for t in raw_data["behavior_tags"]]
    }


@mcp.tool()
async def get_user_tags(
        user_id: str
) -> str:
    """
    Name: 获取用户标签
    Description: 根据用户ID查询其所有标签及权重
    Args:
        user_id: 用户唯一标识（格式：UID_xxxx）
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{TAG_API_ENDPOINT}/users/{user_id}/tags",
                headers={"Authorization": f"Bearer {TAG_API_KEY}"}
            )
            response.raise_for_status()

        formatted_data = _format_tag_data(response.json())
        return wrap_sse_response({
            "status": "success",
            "data": formatted_data
        })

    except httpx.HTTPStatusError as e:
        return wrap_sse_response({"error": f"HTTP error: {e.response.status_code}"})
    except Exception as e:
        return wrap_sse_response({"error": str(e)})


@mcp.tool()
async def create_dynamic_tag(
        tag_definition: dict
) -> str:
    """
    Name: 创建动态标签
    Description: 根据规则定义创建新的动态标签
    Args:
        tag_definition: 标签定义 {
            "name": "高净值用户",
            "rules": [
                {"field": "total_assets", "op": ">", "value": 1000000},
                {"field": "active_days", "op": ">=", "value": 30}
            ]
        }
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{TAG_API_ENDPOINT}/tags",
                json=tag_definition,
                headers={"Authorization": f"Bearer {TAG_API_KEY}"}
            )
            response.raise_for_status()

            return wrap_sse_response({
                "status": "created",
                "tag_id": response.json()["id"]
            })
    except Exception as e:
        return wrap_sse_response({"error": str(e)})


@mcp.tool()
async def analyze_user_segment(
        segment_criteria: dict
) -> str:
    """
    Name: 用户分群分析
    Description: 根据标签组合进行用户分群分析
    Args:
        segment_criteria: 分群条件 {
            "required_tags": ["高净值用户", "活跃用户"],
            "excluded_tags": ["风险用户"]
        }
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.request("GET",
                f"{TAG_API_ENDPOINT}/analytics/segment",
                data=json.dumps(segment_criteria, ensure_ascii=False), 
                headers={"Authorization": f"Bearer {TAG_API_KEY}", "Content-Type": "application/json",}
            )
            response.raise_for_status()

            return wrap_sse_response({
                "segment_size": response.json()["count"],
                "user_sample": response.json()["sample_users"]
            })
    except Exception as e:
        return wrap_sse_response({"error": str(e)})

def main():
    # 三种不同的模式 ["stdio", "sse", "streamable-http"]
    mcp.run(transport="sse")
# 开发测试使用选择STDIO，生产部署使用sse、streamable-http模式；
# 需要特别注意：如果SSE模式的MCP服务器中途重启，而MCP客户端没有重新建立/sse流连接，此时当客户端再次向服务器发送请求时就会发生错误。这种情况下，只需在MCP客户端点击刷新按钮或执行类似的重建连接操作即可解决。

if __name__ == "__main__":
    main()

'''
sse模式的客户端请求
# 客户端请求示例如下：
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient 

async def call_tool(s, session, t, a):
    async with MultiServerMCPClient(s) as client:
        s = await client.sessions[session].call_tool(t, a)
        return s

servers = {'math': {'url': 'http://192.168.3.34:8003/sse', 'transport': 'sse'}}
mcp_server= 'math'
mcp_tool= 'get_user_tags'
tool_params= {'user_id': 'UID_1001'}
res = asyncio.run(call_tool(servers, mcp_server, mcp_tool, tool_params))
[content.text for content in res.content]
Out[10]: ['{"status": "success", "data": {"user_id": "UID_1001", "base_tags": [{"name": "高净值用户", "weight": 0.9}, {"name": "活跃用户", "weight": 0.85}, {"name": "新用户", "weight": 0.7}, {"name": "高消费用户", "weight": 0.8}, {"name": "低频用户", "weight": 0.6}, {"name": "忠诚用户", "weight": 0.95}, {"name": "潜在流失用户", "weight": 0.5}, {"name": "高价值用户", "weight": 0.9}, {"name": "低价值用户", "weight": 0.4}, {"name": "高活跃度用户", "weight": 0.85}], "behavior_tags": [{"name": "购买频率高", "last_active": "2023-10-01T12:00:00Z"}, {"name": "最近登录", "last_active": "2023-10-02T12:00:00Z"}, {"name": "购买大额商品", "last_active": "2023-10-03T12:00:00Z"}, {"name": "参与活动", "last_active": "2023-10-04T12:00:00Z"}, {"name": "浏览时间长", "last_active": "2023-10-05T12:00:00Z"}, {"name": "多次购买", "last_active": "2023-10-06T12:00:00Z"}, {"name": "高评价用户", "last_active": "2023-10-07T12:00:00Z"}, {"name": "推荐新用户", "last_active": "2023-10-08T12:00:00Z"}, {"name": "多次退货", "last_active": "2023-10-09T12:00:00Z"}, {"name": "多次投诉", "last_active": "2023-10-10T12:00:00Z"}]}}']

#######################################################################################################################
#"streamable-http"模式的客户端请求
from mcp.client.streamable_http import streamablehttp_client
from mcp import ClientSession
# Connect to a streamable HTTP server
async def test_tool():
    # Connect to a streamable HTTP server
    async with streamablehttp_client("http://192.168.3.34:8003/mcp") as (
        read_stream,
        write_stream,
        _,
    ):
        # Create a session using the client streams
        async with ClientSession(read_stream, write_stream) as session:
            # Initialize the connection
            await session.initialize()
            # Call a tool
            tool_result = await session.call_tool("get_user_tags", {'user_id': 'UID_1001'})
            return tool_result
            
res = asyncio.run(test_tool())
[content.text for content in res.content]
['{"status": "success", "data": {"user_id": "UID_1001", "base_tags": [{"name": "高净值用户", "weight": 0.9}, {"name": "活跃用户", "weight": 0.85}, {"name": "新用户", "weight": 0.7}, {"name": "高消费用户", "weight": 0.8}, {"name": "低频用户", "weight": 0.6}, {"name": "忠诚用户", "weight": 0.95}, {"name": "潜在流失用户", "weight": 0.5}, {"name": "高价值用户", "weight": 0.9}, {"name": "低价值用户", "weight": 0.4}, {"name": "高活跃度用户", "weight": 0.85}], "behavior_tags": [{"name": "购买频率高", "last_active": "2023-10-01T12:00:00Z"}, {"name": "最近登录", "last_active": "2023-10-02T12:00:00Z"}, {"name": "购买大额商品", "last_active": "2023-10-03T12:00:00Z"}, {"name": "参与活动", "last_active": "2023-10-04T12:00:00Z"}, {"name": "浏览时间长", "last_active": "2023-10-05T12:00:00Z"}, {"name": "多次购买", "last_active": "2023-10-06T12:00:00Z"}, {"name": "高评价用户", "last_active": "2023-10-07T12:00:00Z"}, {"name": "推荐新用户", "last_active": "2023-10-08T12:00:00Z"}, {"name": "多次退货", "last_active": "2023-10-09T12:00:00Z"}, {"name": "多次投诉", "last_active": "2023-10-10T12:00:00Z"}]}}']

'''

'''
from mcp.server.fastmcp import FastMCP
import os
mcp = FastMCP(port=8002)
@mcp.tool()
def list_desktop_files() -> list:
    """获取当前用户桌面上的所有文件列表（macOS专属实现）"""
    desktop_path = os.path.expanduser("~/Desktop")
    return os.listdir(desktop_path)
@mcp.tool()
def say_hello(name: str) -> str:
    """生成个性化问候语（中英双语版）"""
    return f"🎉 你好 {name}! (Hello {name}!)"
@mcp.resource("config://app_settings")
def get_app_config() -> dict:
    return {"theme": "dark", "language": "zh-CN"}
@mcp.prompt()
def code_review_prompt(code: str) -> str:
    return f"请审查以下代码并指出问题：\n\n{code}"
mcp.run(transport='sse')

'''

# 更多参考资料：https://github.com/modelcontextprotocol/python-sdk

