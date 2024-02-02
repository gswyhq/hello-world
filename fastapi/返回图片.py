

from fastapi import FastAPI
from fastapi import FastAPI, File, UploadFile, Response
from fastapi.responses import StreamingResponse
import hashlib
from aiopathlib import AsyncPath  # pip install aiopathlib
from pathlib import Path


#  返回图片
@app.post("/downloadfile2/")
async def download_files_stream():
    file_like = open('as.png', mode="rb")
    return StreamingResponse(file_like, media_type="image/jpg")


@app.get("/show")
async def root():
    html_content = """
    <html>
        <head>
            <title>元素识别服务</title>
        </head>
        <body>
            <h1>请上传文件(png、jpg格式)</h1>
            <form action="/show-image/" method="post" enctype="multipart/form-data">
                <input type="file" name="myfile" />
                <input type="submit" />
            </form>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)


BASE_DIR = Path(__file__).resolve().parent
LIMIT_SIZE = 5 * 1024 * 1024  # 5M
MEDIA_ROOT = BASE_DIR / 'static'
if not MEDIA_ROOT.exists():
    MEDIA_ROOT.mkdir()

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

#  返回图片
@app.post("/show-image")
async def show_image(request: Request, myfile: UploadFile = File(...)):
    """单文件上传（不能大于5M）"""
    bytes_contents = await myfile.read()
    if len(bytes_contents) > LIMIT_SIZE:
        raise HTTPException(status_code=400, detail="每个文件都不能大于5M")

    name = hashlib.md5(bytes_contents).hexdigest() + ".png"  # 使用md5作为文件名，以免同一个文件多次写入
    subpath = "static/uploads"

    if not (folder := BASE_DIR / subpath).exists():
        await AsyncPath(folder).mkdir(parents=True)
    if not (fpath := folder / name).exists():
        await AsyncPath(fpath).write_bytes(bytes_contents)
    save_path = os.path.abspath(fpath)
    logger.info(f'上传文件成功：{save_path}, {myfile.filename}')
    save_file, result = parser_save_file(save_path)
    file_like = open(save_file, mode="rb")
    return StreamingResponse(file_like, media_type="image/jpg")


