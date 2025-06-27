#!/usr/bin/env python
# coding=utf-8
# tag_platform_server/main.py

# 为了演示标签平台的MCP服务，故而先写个标签平台的服务

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
import json
import os
import hashlib, uuid
from datetime import datetime

app = FastAPI(root_path="/v1")

DEFAULT_API_KEY = hashlib.md5(str(uuid.uuid1()).encode()).hexdigest()
DEFAULT_API_KEY = "b5b3db94f086bb2a54154ba4244402be"
# 从环境变量中获取API密钥
TAG_API_KEY = os.getenv("TAG_API_KEY", DEFAULT_API_KEY)

print(f"TAG_API_KEY={TAG_API_KEY}")

# 允许跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 模拟数据库中的用户标签数据
users_tags_db = {
    "UID_1001": {
        "user_id": "UID_1001",
        "basic_tags": [
            {"name": "高净值用户", "score": 0.9},
            {"name": "活跃用户", "score": 0.85},
            {"name": "新用户", "score": 0.7},
            {"name": "高消费用户", "score": 0.8},
            {"name": "低频用户", "score": 0.6},
            {"name": "忠诚用户", "score": 0.95},
            {"name": "潜在流失用户", "score": 0.5},
            {"name": "高价值用户", "score": 0.9},
            {"name": "低价值用户", "score": 0.4},
            {"name": "高活跃度用户", "score": 0.85}
        ],
        "behavior_tags": [
            {"tag_name": "购买频率高", "update_time": "2023-10-01T12:00:00Z"},
            {"tag_name": "最近登录", "update_time": "2023-10-02T12:00:00Z"},
            {"tag_name": "购买大额商品", "update_time": "2023-10-03T12:00:00Z"},
            {"tag_name": "参与活动", "update_time": "2023-10-04T12:00:00Z"},
            {"tag_name": "浏览时间长", "update_time": "2023-10-05T12:00:00Z"},
            {"tag_name": "多次购买", "update_time": "2023-10-06T12:00:00Z"},
            {"tag_name": "高评价用户", "update_time": "2023-10-07T12:00:00Z"},
            {"tag_name": "推荐新用户", "update_time": "2023-10-08T12:00:00Z"},
            {"tag_name": "多次退货", "update_time": "2023-10-09T12:00:00Z"},
            {"tag_name": "多次投诉", "update_time": "2023-10-10T12:00:00Z"}
        ]
    }
}

# 模拟数据库中的动态标签数据
dynamic_tags_db = {
    "TAG_123456": {
        "id": "TAG_123456",
        "name": "活跃消费者",
        "rules": [
            {"field": "purchase_freq", "op": ">", "value": 5},
            {"field": "last_active", "op": ">=", "value": "2024-01-01"}
        ],
        "created_at": datetime.now().isoformat()
    }
}

# 模拟数据库中的用户分群数据
segmentation_db = {
    "sample_users": [
        {"user_id": "UID_1001", "tags": ["高净值用户", "活跃用户"]},
        {"user_id": "UID_1002", "tags": ["高净值用户", "活跃用户"]},
        {"user_id": "UID_1003", "tags": ["高净值用户", "活跃用户"]},
        {"user_id": "UID_1004", "tags": ["高净值用户", "活跃用户"]},
        {"user_id": "UID_1005", "tags": ["高净值用户", "活跃用户"]},
        {"user_id": "UID_1006", "tags": ["高净值用户", "活跃用户"]},
        {"user_id": "UID_1007", "tags": ["高净值用户", "活跃用户"]},
        {"user_id": "UID_1008", "tags": ["高净值用户", "活跃用户"]},
        {"user_id": "UID_1009", "tags": ["高净值用户", "活跃用户"]},
        {"user_id": "UID_1010", "tags": ["高净值用户", "活跃用户"]}
    ],
    "count": 1000
}

class UserTagResponse(BaseModel):
    user_id: str
    base_tags: List[Dict[str, Any]]
    behavior_tags: List[Dict[str, Any]]

class CreateDynamicTagRequest(BaseModel):
    name: str
    rules: List[Dict[str, Any]]

class CreateDynamicTagResponse(BaseModel):
    id: str
    name: str
    rules: List[Dict[str, Any]]
    created_at: str

class SegmentCriteria(BaseModel):
    required_tags: List[str]
    excluded_tags: List[str] = []

class SegmentAnalysisResponse(BaseModel):
    segment_size: int
    user_sample: List[Dict[str, Any]]

def get_authorization_key(request: Request) -> Optional[str]:
    """获取请求中的Authorization密钥"""
    auth_header = request.headers.get("Authorization", "")
    if not auth_header:
        return None
    # 假设Authorization格式为 "Bearer <api_key>"
    parts = auth_header.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    return parts[1]

@app.get("/users/{user_id}/tags")
async def get_user_tags(user_id: str, request: Request):
    """
    获取用户标签
    """
    # 验证Authorization密钥
    api_key = get_authorization_key(request)
    if api_key != TAG_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    if user_id not in users_tags_db:
        raise HTTPException(status_code=404, detail="User not found")
    return users_tags_db[user_id]

@app.post("/tags")
async def create_dynamic_tag(tag_definition: CreateDynamicTagRequest, request: Request):
    """
    创建动态标签
    """
    # 验证Authorization密钥
    api_key = get_authorization_key(request)
    if api_key != TAG_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    tag_id = f"TAG_{len(dynamic_tags_db) + 1}"
    tag = {
        "id": tag_id,
        "name": tag_definition.name,
        "rules": tag_definition.rules,
        "created_at": datetime.now().isoformat()
    }
    dynamic_tags_db[tag_id] = tag
    return CreateDynamicTagResponse(**tag)

@app.get("/analytics/segment")
async def analyze_user_segment(request: Request, required_tags: List[str], excluded_tags: List[str] = []):
    """
    用户分群分析
    """
    # 验证Authorization密钥
    api_key = get_authorization_key(request)
    if api_key != TAG_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    # 模拟分群逻辑
    segment = segmentation_db.copy()
    return segment

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

if __name__ == "__main__":
    main()
    
'''
获取用户标签（包含 Authorization 头）：
curl -XGET http://localhost:8001/v1/users/UID_1001/tags \
-H "Authorization: Bearer b5b3db94f086bb2a54154ba4244402be"

创建动态标签（包含 Authorization 头）：
curl -XPOST http://localhost:8001/v1/tags \
-H "Content-Type: application/json" \
-H "Authorization: Bearer b5b3db94f086bb2a54154ba4244402be" \
-d '{
    "name": "活跃消费者",
    "rules": [
        {"field": "purchase_freq", "op": ">", "value": 5},
        {"field": "last_active", "op": ">=", "value": "2024-01-01"}
    ]
}'

用户分群分析（包含 Authorization 头）：
curl -XGET http://localhost:8001/v1/analytics/segment \
-H "Content-Type: application/json" \
-H "Authorization: Bearer b5b3db94f086bb2a54154ba4244402be" \
-d '{
    "required_tags": ["高净值用户", "活跃用户"],
    "excluded_tags": ["风险用户"]
}'
'''