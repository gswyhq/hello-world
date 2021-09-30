from typing import List

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse

app = FastAPI()


@app.post("/files/")
async def create_files(files: List[bytes] = File(...)):
    file = files[0]
    with open("save_file.txt", 'wb')as f:
        f.write(file)
    return {"file_size": len(file)}


@app.post("/uploadfiles/")
async def create_upload_files(files: List[UploadFile] = File(...)):
    for file in files:
        print("filename: {}".format(file.filename))
        print("content_type: {}".format(file.content_type))
        bytes_contents = await file.read()
        print(bytes_contents)
    return {"filenames": [file.filename for file in files]}


@app.get("/")
async def main():
    content = """
<body>
<form action="/files/" enctype="multipart/form-data" method="post">
<input name="files" type="file" multiple>
<input type="submit">
</form>
<form action="/uploadfiles/" enctype="multipart/form-data" method="post">
<input name="files" type="file" multiple>
<input type="submit">
</form>
</body>
    """
    return HTMLResponse(content=content)


def main():
    # 用uvicorn.run的话，python3 main.py启动即可；
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port="8000", reload=False, log_level="info")

    # 或者
    # uvicorn.run("main:app", host="0.0.0.0", port="8001", reload=False, log_level="info")

    # 若不用uvicorn.run，则需用下面命令启动
    # uvicorn main:app --reload --host 192.XXX.XXX --port 8001

if __name__ == '__main__':
    main()

