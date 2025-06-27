#!/usr/bin/env python
# coding=utf-8
import traceback
import zlib
#########################################################################################################################
import gradio as gr
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
import httpx, gzip
from fastapi.staticfiles import StaticFiles

app = FastAPI(root_path="")

def format_header_key(key):
    # 将键按中划线分割
    if isinstance(key, bytes):
        return format_header_key(key.decode('utf-8')).encode('utf-8')
    parts = key.split('-')
    # 对每个部分应用 str.title() 方法
    formatted_parts = [part.title() for part in parts]
    # 用中划线连接起来
    return '-'.join(formatted_parts)


@app.api_route("/{proxy_path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
async def global_proxy(request: Request, proxy_path: str):
    EXTERNAL_BASE_URL = "http://192.168.3.34:8080"
    print(f"proxy_path={proxy_path}")
    target_url = f"{EXTERNAL_BASE_URL}/{proxy_path.rstrip('/')}"

    # 清理请求头（关键修改）
    headers = {
        k: v for k, v in request.headers.items()
        if k.lower() not in [
            'content-length',
        ]
    }
    print(f"headers: {headers}")
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            # 构建请求时禁用自动解压
            req = client.build_request(
                method=request.method,
                url=target_url,
                headers=headers,
                params=request.query_params,
                content=await request.body(),
            )
            # 关键修改：禁用自动解压和分块传输
            response = await client.send(req, stream=False)

            # 强制修正响应头
            response_headers = dict(response.headers)
            print(f"response_headers: {response_headers}")

            # 仅移除服务器信息类头
            excluded_headers = ['server', 'date', 'content-length', 'content-encoding']
            response_headers = {
                k: v for k, v in response_headers.items()
                if k.lower() not in excluded_headers
            }

            # 直接透传原始响应
            return Response(
                content=response.content,
                status_code=response.status_code,
                headers=response_headers
            )

    except (httpx.ConnectError, httpx.ReadTimeout, httpx.ReadError) as e:
        print(f"后端服务通信失败: {e}")
        print('错误详情：', traceback.format_exc())
        return Response(
            content=b'Upstream service unavailable',
            status_code=502,
            headers={'Content-Type': 'text/plain'}
        )

# 原 Gradio 挂载逻辑
with gr.Blocks() as demo:
    gr.HTML('''<iframe
  src="/ui/chat/fb7b4eb8dc90c18b"
  style="width: 100%; height: 800px;"
  frameborder="0"
  allow="microphone">
</iframe>
''')


# 访问本地静态文件
app.mount("/output_splits", StaticFiles(directory="output_splits"), name="static")

app = gr.mount_gradio_app(app, demo, path="", auth=('admin', '123456'))


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

# 访问 http://192.168.3.34:7860/ui/chat/fb7b4eb8dc90c18b 中介转访问：http://192.168.3.34:8080/ui/chat/fb7b4eb8dc90c18b
if __name__ == "__main__":
    main()

