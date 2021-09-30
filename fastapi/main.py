from fastapi import FastAPI
import time
import uvicorn
from starlette import status

app = FastAPI()


@app.get("/")
def read_root():
    time.sleep(10)
    return {"Hello": "你好"}


@app.get("/items/{item_id}", status_code=status.HTTP_200_OK)
def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}

def main():
    # 用uvicorn.run的话，python3 main.py启动即可；
    uvicorn.run(app, host="0.0.0.0", port="8000", reload=False, log_level="info")

    # 或者
    # uvicorn.run("main:app", host="0.0.0.0", port="8001", reload=False, log_level="info")

    # 若不用uvicorn.run，则需用下面命令启动
    # uvicorn main:app --reload --host 192.XXX.XXX --port 8001

if __name__ == '__main__':
    main()

