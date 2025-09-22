import gradio as gr
import soundfile as sf
from voxcpm import VoxCPM
import os
import tempfile
import logging
import time
import base64
from logging import StreamHandler
from logging.handlers import QueueHandler, QueueListener, TimedRotatingFileHandler
from multiprocessing import Queue
from fastapi import FastAPI, Depends, Request, UploadFile, File, HTTPException, status
from fastapi.responses import StreamingResponse
from fastapi.responses import FileResponse
from fastapi import Form
from pydantic import BaseModel, ConfigDict
from typing import Dict, Any, List, Optional
import logging
import traceback
import os


# 创建日志记录器
logger = logging.getLogger('my_logger')
logger.setLevel(logging.DEBUG)

true = True
false = False
null = None
CONNECT_TIMEOUT, READ_TIMEOUT = 600, 600

# 创建队列
log_queue = Queue()

# 创建处理器，每天生成一个新文件，保留7天的日志
file_handler = TimedRotatingFileHandler(
    'logs/app.log',  # 日志文件名
    when='D',    # 按天分割
    interval=1,  # 每隔1天分割
    backupCount=3650,  # 保留7天的日志
    encoding='utf-8' # 设置文件编码为UTF-8
)

# 创建终端处理器（新增部分）
console_handler = StreamHandler()

# 设置统一的日志格式
formatter = logging.Formatter('%(asctime)s - %(levelname)s - [Process: %(process)d] [Thread: %(thread)d] - [%(filename)s:%(lineno)d] %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)  # 终端使用相同格式

# 创建队列处理器
queue_handler = QueueHandler(log_queue)

# 将队列处理器添加到记录器
logger.addHandler(queue_handler)

# 创建队列监听器
listener = QueueListener(log_queue, file_handler, console_handler)
listener.start()

# 记录日志
logger.info('This is a log message.')

# 加载 VoxCPM 模型(https://hf-mirror.com/openbmb/VoxCPM-0.5B)
model = VoxCPM.from_pretrained("/root/VoxCPM-0.5B", load_denoiser=False)

DEFAULT_VOICE_DICT = {
    "女声": ["/root/VoxCPM-0.5B/examples/617d64a7a0e5.wav", """你好，我是***的，你这边呢是我们的老客户了，相信你也是听说过惠民保的吧，那这个哈，就是咱们福建当地推出来的补充医疗保障，是政府力推的公益性产品，就只要129元就有350万元的保障，而且全家老少都可以保，带病参保也是可以赔的，那目前这边呢，已经是有几十万人呢都保了，今天就是通知你参保的，我已经是给你发短信了，你也赶急点击了解一下吧"""],
    "童声": ["/root/VoxCPM-0.5B/examples/tmpo6j6gkx3.wav", '山行，唐，杜牧，远上寒山石径斜，白云生处有人家，停车坐爱枫林晚，霜叶红于二月花'],
    "alloy": ["/root/VoxCPM-0.5B/examples/tmpceuon29a.wav", '你好，这里是语音播放测试'],
    "echo": ["/root/VoxCPM-0.5B/examples/tmpkagzsim4.wav", '你好，这里是语音播放测试'],
    "fable": ["/root/VoxCPM-0.5B/examples/tmpxhchdxho.wav", '你好，这里是语音播放测试'],
    "onyx": ["/root/VoxCPM-0.5B/examples/tmpfm9nrqc5.wav", '你好，这里是语音播放测试'],
    "nova": ["/root/VoxCPM-0.5B/examples/tmp8d259tds.wav", '你好，这里是语音播放测试'],
    "shimmer": ["/root/VoxCPM-0.5B/examples/tmp12w67ulu.wav", '你好，这里是语音播放测试'],
}

# 生成语音的函数
def generate_audio(prompt_audio, prompt_text, target_text):
    logger.info("✅ 接收到新的语音合成请求")

    # 判断是否为克隆模式
    if prompt_audio:
        logger.info("🎙️ 进入声音克隆模式")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
            if isinstance(prompt_audio, str):
                prompt_wav_path = prompt_audio
            else:
                prompt_audio.save(tmpfile.name)
                prompt_wav_path = tmpfile.name
            logger.info(f"📁 参考音频已保存至: {prompt_wav_path}")

        # 生成语音
        wav = model.generate(
            text=target_text,
            prompt_wav_path=prompt_wav_path,
            prompt_text=prompt_text,
            cfg_value=2.0,
            inference_timesteps=10,
            normalize=True,
            denoise=False,
            retry_badcase=True,
            retry_badcase_max_times=3,
            retry_badcase_ratio_threshold=6.0
        )
        logger.info("🔊 开始保存克隆语音文件")
    else:
        logger.info("🎙️ 进入普通语音合成模式")
        # 普通语音合成模式
        wav = model.generate(
            text=target_text,
            prompt_wav_path=None,
            prompt_text=None,
            cfg_value=2.0,
            inference_timesteps=10,
            normalize=True,
            denoise=False,
            retry_badcase=True,
            retry_badcase_max_times=3,
            retry_badcase_ratio_threshold=6.0
        )
        logger.info("🔊 开始保存普通语音文件")

    # 保存生成的语音到临时文件
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
        sf.write(tmpfile.name, wav, 16000)
        output_path = tmpfile.name
        logger.info(f"✅ 语音合成完成，文件保存至: {output_path}")

    return output_path

# Gradio 界面定义
def voice_cloning_interface(prompt_audio, prompt_text, target_text):
    if not target_text:
        return "❌ 请输入需要合成的文本。"

    logger.info("📝 开始处理语音合成请求")
    output_path = generate_audio(prompt_audio, prompt_text, target_text)
    logger.info("🎉 语音合成流程结束")
    return output_path

# 创建 Gradio 界面
demo = gr.Interface(
    fn=voice_cloning_interface,
    inputs=[
        gr.Audio(type="filepath", label="上传参考音频（.wav，可选）", value=DEFAULT_VOICE_DICT['女声'][0]),
        gr.Textbox(label="参考音频对应的文本（可选）", value=DEFAULT_VOICE_DICT['女声'][1], lines=3),
        gr.Textbox(label="需要合成的文本", lines=3, value='山行，唐，杜牧，远上寒山石径斜，白云生处有人家，停车坐爱枫林晚，霜叶红于二月花')
    ],
    outputs=gr.Audio(label="生成的语音", type="filepath"),
    title="🎙️ VoxCPM 语音合成服务",
    description="输入文本，可选上传参考音频和文本，即可生成语音。"
)


# 启用 Gradio 的队列机制（保留你的配置）
demo.queue(
    status_update_rate=1,
    default_concurrency_limit=120,
    max_size=120
)

app = FastAPI(root_path="/ai")

# TTS Model
class TTSRequest(BaseModel):
    model: str
    input: str
    voice: str
    response_format: Optional[str] = "wav"
    speed: Optional[float] = 1.0

# Transcription Model
class TranscriptionRequest(BaseModel):
    model: str
    language: Optional[str] = None
    prompt: Optional[str] = None
    response_format: Optional[str] = "json"
    temperature: Optional[float] = 0.0
    timestamp_granularities: Optional[List[str]] = ["segment"]

# Constants
CHAT_OPENAI_API_KEY = "chat-9sF!2Lm@qW7xZ$423we23e"
MODEL_TOKEN_DICT = {
    "tts-1": "tts-erwweofwehFWEOFJfweff",
    "whisper-1": "whisper-dwhfweWEFWEWFWDWDDdfwf"
}

# API Key 验证依赖
async def verify_api_key(request: Request):
    api_key = request.headers.get("Authorization")
    logger.info(f"api-key={api_key}")
    if api_key is None or not api_key.startswith("Bearer "):
        logger.error(f"认证无效：{api_key}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid Authorization header"
        )
    token = api_key.split("Bearer ")[1].strip()
    try:
        item_data = await request.json()
        logger.info(f"item_data: {item_data}")
        model = item_data.get("model")
    except:
        try:
            form_data = await request.form()
            model = form_data.get("model")
        except Exception as e:
            logger.error(f"解析表单失败: {e}")
            model = None

    logger.info(f"request.url.path: {request.url.path}")

    # 如果是 /v1/models 接口，允许任意一个有效的 API Key
    if request.url.path.startswith("/v1/models"):
        valid_keys = list(MODEL_TOKEN_DICT.values()) + [CHAT_OPENAI_API_KEY]
        if token not in valid_keys:
            logger.error(f"认证无效：{api_key}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API Key"
            )
        return token

    expected_api_key = MODEL_TOKEN_DICT.get(model, CHAT_OPENAI_API_KEY)
    if token != expected_api_key:
        logger.error(f"认证无效：{api_key}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key"
        )
    return token

@app.get("/v1/models")
async def list_models(token: str = Depends(verify_api_key)):
    logger.info("获取模型列表请求")
    # 模拟模型列表数据
    models = [
        {
            "id": "tts-1",
            "object": "model",
            "owned_by": "organization-owner",
            "permission": [
                {
                    "id": "perm-1",
                    "object": "model_permission",
                    "created": int(time.time()),
                    "allow_create_engine": False,
                    "allow_sampling": True,
                    "allow_logprobs": True,
                    "allow_search_indices": False,
                    "allow_view": True,
                    "allow_fine_tuning": False,
                    "organization": "*",
                    "group": None,
                    "is_blocking": False
                }
            ]
        },
        {
            "id": "whisper-1",
            "object": "model",
            "owned_by": "organization-owner",
            "permission": [
                {
                    "id": "perm-2",
                    "object": "model_permission",
                    "created": int(time.time()),
                    "allow_create_engine": False,
                    "allow_sampling": True,
                    "allow_logprobs": True,
                    "allow_search_indices": False,
                    "allow_view": True,
                    "allow_fine_tuning": False,
                    "organization": "*",
                    "group": None,
                    "is_blocking": False
                }
            ]
        }
    ]
    return {
        "object": "list",
        "data": models
    }

@app.get("/v1/models/{model_id}")
async def get_model(model_id: str, token: str = Depends(verify_api_key)):
    logger.info(f"获取模型详情请求: {model_id}")
    # 模拟模型详情数据
    models = {
        "tts-1": {
            "id": "tts-1",
            "object": "model",
            "owned_by": "organization-owner",
            "permission": [
                {
                    "id": "perm-1",
                    "object": "model_permission",
                    "created": int(time.time()),
                    "allow_create_engine": False,
                    "allow_sampling": True,
                    "allow_logprobs": True,
                    "allow_search_indices": False,
                    "allow_view": True,
                    "allow_fine_tuning": False,
                    "organization": "*",
                    "group": None,
                    "is_blocking": False
                }
            ]
        },
        "whisper-1": {
            "id": "whisper-1",
            "object": "model",
            "owned_by": "organization-owner",
            "permission": [
                {
                    "id": "perm-2",
                    "object": "model_permission",
                    "created": int(time.time()),
                    "allow_create_engine": False,
                    "allow_sampling": True,
                    "allow_logprobs": True,
                    "allow_search_indices": False,
                    "allow_view": True,
                    "allow_fine_tuning": False,
                    "organization": "*",
                    "group": None,
                    "is_blocking": False
                }
            ]
        }
    }

    if model_id not in models:
        raise HTTPException(status_code=404, detail="Model not found")

    return models[model_id]

@app.post("/v1/audio/speech")
async def text_to_speech(tts_request: TTSRequest, request: Request, token: str = Depends(verify_api_key)):
    logger.info(f"TTS 请求: {tts_request.model_dump()}")
    # 模拟生成语音文件
    voice = tts_request.voice
    prompt_audio, prompt_text = None, None
    if voice in DEFAULT_VOICE_DICT.keys():
        prompt_audio, prompt_text = DEFAULT_VOICE_DICT[voice]
        if not os.path.isfile(prompt_audio):
            prompt_audio, prompt_text = None, None
    target_text = tts_request.input
    if prompt_text == target_text:
        output_path = prompt_audio
    else:
        output_path = generate_audio(prompt_audio, prompt_text, target_text)
    # 检查文件是否存在
    if not os.path.exists(output_path):
        raise HTTPException(status_code=500, detail="音频文件生成失败")

    return FileResponse(output_path, media_type="audio/wav")

########################################################################################################################
# 加载语音识别、语音转录模型

import os
import gradio as gr
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset
USERNAME = os.getenv("USERNAME")

# 模型来源：https://hf-mirror.com/openai/whisper-large-v3-turbo
# 数据集来源：https://hf-mirror.com/datasets/distil-whisper/librispeech_long/tree/main

# 使用putenv函数设置新的PATH环境变量
# https://www.gyan.dev/ffmpeg/builds/packages/ffmpeg-7.0.2-full_build.7z
#os.environ['PATH'] = f'D:\\Users\\{USERNAME}\\ffmpeg-7.1\\bin\\;' + os.getenv('PATH')
#print(os.getenv('PATH'))  # 输出新的PATH环境变量

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# model_id = "openai/whisper-large-v3-turbo"
model_id = rf"/root/whisper-large-v3-turbo"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id, language="zh")

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)


@app.post("/v1/audio/transcriptions")
async def transcribe_audio(
    file: UploadFile = File(...),
    model: str = Form("whisper-1"),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    response_format: Optional[str] = Form("json"),
    temperature: Optional[float] = Form(0.0),
    timestamp_granularities: Optional[List[str]] = Form(None),
    request: Request = None,
    token: str = Depends(verify_api_key)
):
    logger.info(f"Transcription 请求: model={model}, language={language}, response_format={response_format}")
    logger.info(f"文件名: {file.filename}, 文件类型: {file.content_type}")

    # 读取上传的文件内容
    contents = await file.read()
    file_path = f"/tmp/{file.filename}"
    with open(file_path, "wb") as f:
        f.write(contents)

    # 调用 Whisper 模型进行转录
    result = pipe(file_path)
    logger.info(f"{file.filename} 转录的结果：{result}")
    # 模拟转录逻辑
    # 实际应读取文件内容并调用 Whisper 模型进行转录
    # return {       "text": "Imagine the wildest idea that you've ever had, and you're curious about how it might scale to something that's a 100, a 1,000 times bigger. This is a place where you can get to do that."}
    return result


# 挂载 Gradio 应用到 FastAPI
app = gr.mount_gradio_app(app, demo, path="", mcp_server=True)

# 启动服务
if __name__ == "__main__":
    logger.info("🚀 正在启动 Gradio 服务...")
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
    logger.info("✅ Gradio 服务已启动，访问 http://localhost:7860")


'''
curl http://localhost:8723/v1/audio/speech \
  -H "Authorization: Bearer tts-erwweofwehFWEOFJfweff" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-1",
    "input": "山行，唐，杜牧，远上寒山石径斜，白云生处有人家，停车坐爱枫林晚，霜叶红于二月花",
    "voice": "女声"
  }' \
  --output speech.wav

curl -X POST http://localhost:8723/v1/audio/transcriptions \
  -H "Authorization: Bearer whisper-dwhfweWEFWEWFWDWDDdfwf" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@tmpo6j6gkx3.wav" \
  -F "model=whisper-1" \
  -F "language=zh" \
  -F "response_format=json"
  
'''

