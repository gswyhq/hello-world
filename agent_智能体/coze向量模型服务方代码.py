from sentence_transformers import SentenceTransformer
import os
import requests
import requests
import json
import logging
from logging import StreamHandler
from logging.handlers import QueueHandler, QueueListener, TimedRotatingFileHandler
from multiprocessing import Queue
from requests.exceptions import RequestException
from fastapi import FastAPI, Request, Response, status, HTTPException, Depends
from fastapi.responses import JSONResponse, StreamingResponse
import httpx, gzip
from fastapi.staticfiles import StaticFiles


import logging
from logging.handlers import TimedRotatingFileHandler

# 创建日志记录器
logger = logging.getLogger('my_logger')
logger.setLevel(logging.DEBUG)

true = True
false = False
null = None
CONNECT_TIMEOUT, READ_TIMEOUT = 600, 600

# 创建队列
log_queue = Queue()

# 创建处理器，每天生成一个新文件，保留7天的日志
file_handler = TimedRotatingFileHandler(
    'logs/app.log',  # 日志文件名
    when='D',    # 按天分割
    interval=1,  # 每隔1天分割
    backupCount=3650,  # 保留n天的日志
    encoding='utf-8' # 设置文件编码为UTF-8
)

# 创建终端处理器（新增部分）
console_handler = StreamHandler()

# 设置统一的日志格式
formatter = logging.Formatter('%(asctime)s - %(levelname)s - [Process: %(process)d] [Thread: %(thread)d] - [%(filename)s:%(lineno)d] %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)  # 终端使用相同格式

# 创建队列处理器
queue_handler = QueueHandler(log_queue)

# 将队列处理器添加到记录器
logger.addHandler(queue_handler)

# 创建队列监听器
listener = QueueListener(log_queue, file_handler, console_handler)
listener.start()

# 记录日志
logger.info('This is a log message.')


model_dir = rf"/root/bge-small-zh-v1.5"
model = SentenceTransformer(model_dir)

import gradio as gr

with gr.Blocks() as demo:
    gr.Markdown("向量模型")

# 启用 Gradio 的队列机制（保留你的配置）
demo.queue(
    status_update_rate=1,
    default_concurrency_limit=120,
    max_size=120
)

from fastapi import FastAPI, HTTPException, Depends, Request, status
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import uuid
import time
from fastapi.middleware import Middleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
app = FastAPI()



def achat(model_name, stream, messages, generateParam=None):
    headers = {
        "content-type": "application/json",
    }

    try:
        response = requests.post(STREAM_CHAT_URL,
            json=chat_gpt_dialog,
            headers=headers,
            stream=True,
            timeout=(CONNECT_TIMEOUT, READ_TIMEOUT)  # 连接超时和读取超时均为600秒
        )

        answerStr = ''
        response.raise_for_status()

        if stream:
            # 模拟流式响应生成器
            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8')
                    logger.info(f"请求结果：{line_str}", )
                    answer = json.loads(line_str[5:])['text'][0]
                    json_data = {'choices': [{'delta': {'content': answer, 'role': 'assistant'}}]}
                    yield f"data: {json.dumps(json_data, ensure_ascii=False)}\n\n"
        else:
            # 逐行读取响应内容
            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8')
                    logger.info(f"请求结果：{line_str}", )
                    answer = json.loads(line_str[5:])['text'][0]
                    answerStr += answer
            logger.info(f"大模型请求总耗时（秒）：{time.time() - start_time}，结果：{answerStr}")
            yield {"code": 0, "msg": "请求成功", "choices": [
                        {
                          "index": 0,
                          "message": {
                            "role": "assistant",
                            "content": answerStr
                          },
                          "finish_reason": "stop",
                        }
                      ],
                    }
    except httpx.RequestError as e:
        logger.error(f"请求失败: {str(e)}")
        logger.error(f"错误详情：{traceback.format_exc()}")
        yield {"code": 1, "msg": f"{e}"}
    except Exception as e:
        logger.error(f"处理响应失败: {str(e)}")
        logger.error(f"错误详情：{traceback.format_exc()}")
        yield {"code": 1, "msg": f"{e}"}

# 参数验证失败了，会触发RequestValidationError异常
# 捕获参数异常
@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(request: Request, exc: RequestValidationError):
    # 尝试解析请求body
    try:
        body = await request.json()
    except json.JSONDecodeError:
        body = "无法解析的请求body"

    logger.info(f"参数不对{request.method} {request.url}, body: {body}") # 可以用日志记录请求信息,方便排错
    return JSONResponse({"code": "400", "message": exc.errors()})


# 自定义中间件：捕获所有请求，包括 404
@app.middleware("http")
async def log_all_requests(request: Request, call_next):
    logger.info(f"请求: {request.method} {request.url}")
    logger.info(f"Headers: {request.headers}")

    try:
        body = await request.body()
        logger.info(f"Body: {body.decode()}")
    except Exception as e:
        logger.warning(f"读取请求体失败: {e}")

    # 调用下一个中间件或路由处理函数
    response = await call_next(request)

    return response


# 捕获 404 的异常处理器（仅在你手动抛出 HTTPException(404) 时触发）
@app.exception_handler(404)
async def not_found_exception_handler(request: Request, exc: HTTPException):
    logger.error(f"404 Not Found: {request.method} {request.url}")
    try:
        body = await request.body()
        logger.error(f"404 Body: {body.decode()}")
    except Exception as e:
        logger.warning(f"读取 404 请求体失败: {e}")
    return JSONResponse(
        status_code=404,
        content={
            "error": {
                "message": "The requested URL was not found on the server.",
                "type": "invalid_request_error",
                "param": None,
                "code": "not_found"
            }
        }
    )


# 模拟 API Key 验证
EMD_API_KEY = "emb-9sF!2Lm@qW7xZ423we23e"


class ChatItem(BaseModel):
    model: str
    messages: list = None
    input: Dict[str, Any] = {}
    parameters: Dict[str, Any] = {}
    extra_params: Dict[str, Any] = {}  # 可选字段，用于捕获额外参数
    max_tokens: int = 8192
    stream: bool = False
    generateParam: Dict = None
    model_config = ConfigDict(ignored_types=(property, classmethod, staticmethod))


# API Key 验证依赖
def verify_api_key(request: Request):
    api_key = request.headers.get("Authorization")
    logger.info(f"api-key={api_key}")
    if api_key is None or not api_key.startswith("Bearer "):
        logger.error(f"认证无效：{api_key}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid Authorization header"
        )
    token = api_key.split("Bearer ")[1].strip()
    if token != CHAT_OPENAI_API_KEY:  # 替换为你的 API Key
        logger.error(f"认证无效：{api_key}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key"
        )
    return token


# 请求体模型
class EmbeddingRequest(BaseModel):
    model: str
    input: List[str]
    encoding_format: Optional[str] = "float"

# 响应体模型
class EmbeddingResponse(BaseModel):
    data: List[Dict]  # 二维数组
    model: str
    object: str
    usage: dict
    id: str

@app.post("/compatible-mode/v1/embeddings")
async def create_embeddings(
    request: EmbeddingRequest,
    api_key: str = Depends(verify_api_key)
):
    # 模拟 token 计算
    input_text_list = request.input
    logger.info(f"输入：{input_text_list}")
    tokens = len(input_text_list)  # 简单的 token 计算方式

    # 生成 embedding
    embeddings = model.encode(input_text_list, normalize_embeddings=True).tolist()

    # 构造响应
    response = EmbeddingResponse(
        data=[ {
              "object": "embedding",
              "embedding": embedding,
              "index": idx
            } for idx, embedding in enumerate(embeddings)],
        model=request.model,
        object="list",
        usage={"prompt_tokens": tokens, "total_tokens": tokens},
        id=str(uuid.uuid4())
    )

    return response


@app.post("/v2/chat/completions")
async def chatV2(item: ChatItem, request: Request, token: str = Depends(verify_api_key)):
    try:
        logger.info("请求模型：{}".format(item.model))
        # 示例返回
        if not MODEL_API_KEY_DICT.get(item.model):
            return {"code": 1, "msg": f"不支持模型：{item.model}, 仅支持：{list(MODEL_API_KEY_DICT.keys())}"}
        # 检查是否是流式请求
        if item.generateParam and item.generateParam.get("stream", False) or item.stream:
            logger.info("流式请求")
            stream_gen = achat(item.model, True, item.messages, generateParam=item.generateParam)
            return StreamingResponse(stream_gen, media_type="text/event-stream")
        else:
            logger.info("非流式请求")
            result = ""
            for chunk in achat(item.model, False, item.messages, generateParam=item.generateParam):
                result = chunk
            return result
    except Exception as e:
        logger.error(f"请求出错：{e}")
        logger.error(f"错误详情：{traceback.format_exc()}")
        return {"code": 1, "msg": f"请求出错：{e}"}

# 挂载 Gradio 应用到 FastAPI
app = gr.mount_gradio_app(app, demo, path="",
                          # auth=("admin", "123456")
                          )

def main():
    # 启动Gradio应用
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()

