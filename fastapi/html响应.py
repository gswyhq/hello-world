
# 使用 HTMLResponse 来从 FastAPI 中直接返回一个 HTML 响应。
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI()


@app.get("/items/", response_class=HTMLResponse)
async def read_items():
    return """
    <html>
        <head>
            <title>Some HTML in here</title>
        </head>
        <body>
            <h1>Look ma! HTML!</h1>
        </body>
    </html>
    """
# 提示
# 参数 response_class 也会用来定义响应的「媒体类型」。
# 在这个例子中，HTTP 头的 Content-Type 会被设置成 text/html。
# 并且在 OpenAPI 文档中也会这样记录。

####################################################################################################################################################
# 返回一个 Response
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI()


@app.get("/items/")
async def read_items():
    html_content = """
    <html>
        <head>
            <title>Some HTML in here</title>
        </head>
        <body>
            <h1>Look ma! HTML!</h1>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content, status_code=200)
# 警告
# 路径操作函数 直接返回的 Response 不会被 OpenAPI 的文档记录（比如，Content-Type 不会被文档记录），并且在自动化交互文档中也是不可见的。


####################################################################################################################################################

# 直接返回 HTMLResponse
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI()


def generate_html_response():
    html_content = """
    <html>
        <head>
            <title>Some HTML in here</title>
        </head>
        <body>
            <h1>Look ma! HTML!</h1>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content, status_code=200)


@app.get("/items/", response_class=HTMLResponse)
async def read_items():
    return generate_html_response()

# 在这个例子中，函数 generate_html_response() 已经生成并返回 Response 对象而不是在 str 中返回 HTML。
# 通过返回函数 generate_html_response() 的调用结果，你已经返回一个重载 FastAPI 默认行为的 Response 对象，
# 但如果你在 response_class 中也传入了 HTMLResponse，FastAPI 会知道如何在 OpenAPI 和交互式文档中使用 text/html 将其文档化为 HTML。

####################################################################################################################################################
# 直接返回 Response
# Response 类接受如下参数：
# 
# content - 一个 str 或者 bytes。
# status_code - 一个 int 类型的 HTTP 状态码。
# headers - 一个由字符串组成的 dict。
# media_type - 一个给出媒体类型的 str，比如 "text/html"。
# FastAPI（实际上是 Starlette）将自动包含 Content-Length 的头。它还将包含一个基于 media_type 的 Content-Type 头，并为文本类型附加一个字符集。


from fastapi import FastAPI, Response

app = FastAPI()


@app.get("/legacy/")
def get_legacy_data():
    data = """<?xml version="1.0"?>
    <shampoo>
    <Header>
        Apply shampoo here.
    </Header>
    <Body>
        You'll have to use soap here.
    </Body>
    </shampoo>
    """
    return Response(content=data, media_type="application/xml")

####################################################################################################################################################
# 接受文本或字节并返回 HTML 响应。
# 
# PlainTextResponse¶
# 接受文本或字节并返回纯文本响应。


from fastapi import FastAPI
from fastapi.responses import PlainTextResponse

app = FastAPI()


@app.get("/", response_class=PlainTextResponse)
async def main():
    return "Hello World"


