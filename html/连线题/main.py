import os.path
import logging.config
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from fastapi.responses import JSONResponse

# 配置日志记录器
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(levelname)s - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S"
        }
    },
    "handlers": {
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": "app.log",
            "maxBytes": 100 * 1024 * 1024,  # 100MB
            "backupCount": 5, # 最多保留5个备份文件
            "encoding": "utf-8",
            "formatter": "default"
        },
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default"
        }
    },
    "loggers": {
        "": {  # 根记录器
            "handlers": ["file", "console"],
            "level": "INFO",
            "propagate": True
        }
    }
}

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)

app = FastAPI()

# 提供静态文件服务
app.mount("/static", StaticFiles(directory="static"), name="static")

# 数据模型
class LineModel(BaseModel):
    questions: list[str]
    english_options: list[str]
    options: list[str]
    relationships: list[list[str]] = []

@app.get("/api/get_units")
async def get_units():
    return ["unit1", "unit2"]

@app.get("/api/questions")
async def get_questions(unit: str = None):
    wav_path = rf"static\output_splits\{unit}"
    if os.path.isdir(wav_path):
        # 获取指定目录下的音频文件列表
        wav_list = os.listdir(wav_path)
        return {
            "questions": wav_list[:4],  # 音频文件列表
            "english_options": ["elephant", "tiger", "zebra", "panda"],
            "options": ["大象", "老虎", "斑马", "熊猫"],
            "relationships": []
        }
    else:
        return JSONResponse(
            content={"error": "Invalid unit"},
            status_code=400
        )

# 提交连线结果
@app.post("/api/submit")
async def submit_lines(request: Request):
    data = await request.json()
    logger.info(f"提交的数据：{data}")
    correct_relationships = {
        "音频1.wav": "elephant",
        "音频2.wav": "tiger",
        "音频3.wav": "zebra",
        "音频4.wav": "panda",
        "elephant": "大象",
        "tiger": "老虎",
        "zebra": "斑马",
        "panda": "熊猫"
    }
    user_relationships = {line[0]: line[1] for line in data["relationships"]}
    score = 0
    for user_key, user_value in user_relationships.items():
        if correct_relationships.get(user_key) == user_value:
            score += 1
    score_percentage = f"{score * 100 / len(correct_relationships):.2f}".replace('.00', '')
    logger.info(f"得分：{score_percentage}")
    return JSONResponse(content={
        "score": score_percentage,
        "correct_relationships": correct_relationships
    })

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()

# 浏览器打开：http://localhost:7860/static/index.html
# 选择测试的单元，开始连线，提交到后台，判断对错及得分；
#
# +--- main.py
# +--- static
# |   +--- index.html
# |   +--- output_splits
# |   |   +--- unit1
# |   |   |   +--- split_1.wav
# |   |   |   +--- split_2.wav
# |   |   |   +--- split_3.wav
# |   |   |   +--- split_4.wav
# |   |   |   +--- split_5.wav
# |   |   +--- unit2
# |   |   |   +--- split_10.wav
# |   |   |   +--- split_6.wav
# |   |   |   +--- split_7.wav
# |   |   |   +--- split_8.wav
# |   |   |   +--- split_9.wav
# |   +--- script.js
# |   +--- style.css
