import secrets

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi import Request
from fastapi.responses import HTMLResponse
from fastapi.responses import RedirectResponse

app = FastAPI()

security = HTTPBasic()


def get_current_username(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = secrets.compare_digest(credentials.username, "stanleyjobson")
    # 使用了secrets.compare_digest(). 将采取相同的时间进行比较stanleyjobsox，防止 stanleyjobson 与 johndoe 的比较 与 stanleyjobson的比较需要的时间多（哪怕是多了微秒级）。
    correct_password = secrets.compare_digest(credentials.password, "swordfish")
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户名或密码错误！",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


@app.get("/users/me")
def read_current_user(username: str = Depends(get_current_username)):
    return {"username": username}

def get_current_username2(request: Request, credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = secrets.compare_digest(credentials.username, "admin")
    # 使用了secrets.compare_digest(). 将采取相同的时间进行比较stanleyjobsox，防止 stanleyjobson 与 johndoe 的比较 与 stanleyjobson的比较需要的时间多（哪怕是多了微秒级）。
    correct_password = secrets.compare_digest(credentials.password, "123456")
    if not (correct_username and correct_password):
        html_content = """
        <html>
            <head>
                <title></title>
            </head>
            <body>
                "<script>alert('用户名或密码错误！')</script>"
            </body>
        </html>
        """
        # media_type = "text/html"
        return HTMLResponse(content=html_content, status_code=200)
    client_host = request.client.host # "127.0.0.1"
    return {"username": credentials.username, 'client_host': client_host}

@app.get("/users/me2")
def read_current_user2(request: Request, username: str = Depends(get_current_username2)):
    client_host = request.client.host
    print('client_host', client_host)
    return username

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

# https://fastapi.tiangolo.com/zh/advanced/security/http-basic-auth/
