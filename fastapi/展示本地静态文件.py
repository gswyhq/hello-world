# encoding:utf-8

import uvicorn
from fastapi import FastAPI, Request, Response, Body
from fastapi.responses import HTMLResponse
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from tts import tts

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/api/tts", response_class=FileResponse)
def get_tts(text: str):
    some_file_path = tts(text)
    return some_file_path
    # return Response(some_file_path, media_type="audio/wav")

@app.post("/api/tts")
def post_tts(
        text: str = Body(..., title="转换为语音的文本", embed=True)
    ):
    some_file_path = tts(text)
    def iterfile():
        with open(some_file_path, mode="rb") as file_like:
            yield from file_like

    return StreamingResponse(iterfile(), media_type="audio/wav")

@app.get("/", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000, log_level="info")

