from fastapi import FastAPI
import time
# import uvicorn 
import asyncio

app = FastAPI()
# 启动时，会报错：TypeError: %d format: a number is required, not str
# 可以替换为自定义路由类，from fastapi import APIRouter，使用方法同：FastAPI
# APIRouter, FastAPI 也可以同时使用：
# router = APIRouter()
# app.include_router(router)

@app.get("/")
async def read_root():
    time.sleep(10) # 异步函数里头同步的调用，会阻塞整个进程; 可使用asyncio.run_in_executor，将同步变为异步。或者改为异步调用（asyncio.sleep(1)）
    # await asyncio.sleep(1)
    return {"Hello": "World"}


@app.get("/items/{item_id}")
async def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}


def main():
    import uvicorn 
    # 用uvicorn.run的话，python3 async_main.py启动即可；
    uvicorn.run(app, host="0.0.0.0", port="8000", reload=False, log_level="info")

    # 或者
    # uvicorn.run("async_main:app", host="0.0.0.0", port="8001", reload=False, log_level="info")

    # 若不用uvicorn.run，则需用下面命令启动
    # uvicorn async_main:app --reload --host 192.XXX.XXX --port 8001


if __name__ == '__main__':
    main()

# 请求 time curl localhost:8000/，会阻塞 curl http://127.0.0.1:8000/items/5?q=%E4%B8%AD%E5%9B%BD2 的请求；

