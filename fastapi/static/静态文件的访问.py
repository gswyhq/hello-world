from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")


templates = Jinja2Templates(directory="templates")


@app.get("/items/{id}", response_class=HTMLResponse)
async def read_item(request: Request, id: str):
    return templates.TemplateResponse("item.html", {"request": request, "id": id})

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

# python 静态文件的访问.py 
# 浏览器打开： http://localhost:8000/items/1

