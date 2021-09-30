
from fastapi import FastAPI, Request

app = FastAPI()
# app = FastAPI(root_path="/api/v1")


@app.get("/app")
def read_main(request: Request):
    print(request.scope.get("root_path"))
    return {"message": "你好，我是主应用！"}


subapi = FastAPI()
#
#
@subapi.get("/sub")
def read_sub():
    return {"message": "来自子应用的问候"}


app.mount("/subapi", subapi)


def main():
    import uvicorn
    # 用uvicorn.run的话，python3 main.py启动即可；
    uvicorn.run(app, host="0.0.0.0", port="8000", reload=False, log_level="info")

    # 或者
    # uvicorn.run("main:app", host="0.0.0.0", port="8001", reload=False, log_level="info")

    # 若不用uvicorn.run，则需用下面命令启动
    # uvicorn main:app --reload --host 192.XXX.XXX --port 8001

if __name__ == '__main__':
    main()

#  curl -XGET localhost:8000/app
# {"message":"你好，我是主应用！"}                                                                                                                                                                                    ✔
# curl -XGET localhost:8000/subapi/sub
# {"message":"来自子应用的问候"}
# 主应用文档地址：http://localhost:8000/docs  , 主应用文档中是看不到子应用文档。
# 子应用文档地址：http://localhost:8000/subapi/docs

