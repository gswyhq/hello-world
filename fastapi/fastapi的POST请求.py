#!/usr/bin/env python
# coding=utf-8

import uvicorn
import json, os
from pydantic import BaseModel
from typing import Optional
from fastapi import FastAPI, Request, Path, Query, Body
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from typing import Any, ClassVar, Dict, Generator, Literal, Set, Tuple, TypeVar, Union
from fastapi import FastAPI, File, UploadFile, Request

app = FastAPI()

# FastAPI框架自带了对请求参数的验证，包括在路径中的参数、请求的参数以及Body中的参数，使用Path提取和验证路径参数；使用Query提取和验证

# 有时候不知道客户端会请求什么参数，这时就需要对参数进行捕获，以便排查问题
# “Post Unprocessable Entity” 是 HTTP 状态码 422 的错误，表示客户端提交的请求无法被服务器处理。在 FastAPI 中，当我们尝试使用 POST 请求发送数据到服务器时，如果请求中的数据无法通过验证，就会返回这个错误。
# “Post Unprocessable Entity” 错误是由于 FastAPI 请求数据无法通过验证而引起的。为了解决这个错误，我们需要确保请求数据的格式、字段和类型与定义的模型匹配，并且可以通过定义的验证规则。

# 参数验证失败了，会触发RequestValidationError异常
# 捕获参数异常
@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(request: Request, exc: RequestValidationError):
    # 尝试解析请求body
    try:
        body = await request.json()
    except json.JSONDecodeError:
        body = "无法解析的请求body"

    print(f"参数不对{request.method} {request.url}, body: {body}") # 可以用日志记录请求信息,方便排错
    return JSONResponse({"code": "400", "message": exc.errors()})

# {'model': 'qwen-plus', 'input': {'messages': [{'role': 'user', 'content': 'Hi there'}]}, 'parameters': {'temperature': 0.0, 'result_format': 'message'}}

class Item(BaseModel):
    model: str
    messages: list = None
    input: Dict[str, Any] = {}
    parameters: Dict[str, Any] = {}
    extra_params: Dict[str, Any] = {}  # 可选字段，用于捕获额外参数

# curl -XPOST http://192.168.3.73:8000/services/aigc/text-generation/generation \
#   -H "Content-Type: application/json" \
#   -d '{
#     "model": "Qwen2.5-0.5B-Instruct",
#     "messages": [{"role": "user", "content": "中国首都在哪里？"}]
#   }'

@app.post("/services/aigc/text-generation/generation")
def dashscope(item: Item):
    '''
    {'model': 'qwen-plus', 'input': {'messages': [{'role': 'user', 'content': 'Hi there'}]}, 'parameters': {'temperature': 0.0, 'result_format': 'message'}}
    '''
    print("item", item)
    # print("kwargs: ", kwargs)
    if item.input.get('messages'):
        item.messages = item.input['messages']
    else:
        item.messages = '中国首都在哪里'
    return {"text": "北京"}

# Path方法获取请求路径里面的参数如 http://127.0.0.1:8000/bar/123
# Query方法获取请求路径后面的查询参数如 http://127.0.0.1:8000/bar?name=xiaoming&age=18
# Body方法获取请求体里面的参数

def main():
    # 用uvicorn.run的话，python3 main.py启动即可；
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("CHAT_PORT") or 8000), reload=False, log_level="info")

    # 或者
    # uvicorn.run("main:app", host="0.0.0.0", port="8001", reload=False, log_level="info")

    # 若不用uvicorn.run，则需用下面命令启动
    # uvicorn main:app --reload --host 192.XXX.XXX --port 8001

if __name__ == '__main__':
    main()
