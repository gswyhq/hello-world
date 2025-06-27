import gradio as gr
import time
from threading import Thread
from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import json
import time
import logging
from logging import StreamHandler
from logging.handlers import TimedRotatingFileHandler

# 创建日志记录器
logger = logging.getLogger('my_logger')
logger.setLevel(logging.DEBUG)

# 创建处理器，每天生成一个新文件，保留7天的日志
file_handler = TimedRotatingFileHandler(
    'app.log',  # 日志文件名
    when='D',    # 按天分割
    interval=1,  # 每隔1天分割
    backupCount=3650,  # 保留7天的日志
    encoding='utf-8' # 设置文件编码为UTF-8
)

# 创建终端处理器（新增部分）
console_handler = StreamHandler()

# 设置统一的日志格式
formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)  # 终端使用相同格式

# 将处理器添加到记录器
logger.addHandler(file_handler)
logger.addHandler(console_handler)  # 新增：添加终端处理器


USERNAME = os.getenv("USERNAME")

model_name = "Qwen/Qwen3-0.6B"
MODEL_PATH = rf"D:\Users\{USERNAME}\data\Qwen3-0.6B"

class DeepSeekModel:
    def __init__(self, model_path="/root/DeepSeek-R1-Distill-Qwen-1.5B"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # 加载tokenizer和模型
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto"
        )

    def generate_response(self, messages, enable_thinking=True, max_new_tokens=32768):
        try:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking
            )
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
            streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True) # 处理流式请求
            generation_kwargs = dict(**model_inputs, streamer=streamer, max_new_tokens=max_new_tokens)
            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()
            generated_text = ""
            for answer in streamer:
                generated_text += answer
                yield answer
            logger.info(f"模型输出结果：{generated_text}")
        except Exception as e:
            print(f"生成回答时发生错误: {str(e)}")
            return ''

# 初始化模型
model = DeepSeekModel(model_path=MODEL_PATH)
# response = model.generate_response(user_input)


def chat(history_size, enable_thinking, history, request: gr.Request):
    user_id = request.session_hash
    message = history[-1][0]
    start_time = time.time()
    logger.info(f"用户ID：{user_id}，请求的问题：{message}, 保留对话轮数：{history_size}")

    logger.info(f"enable_thinking: {enable_thinking}")
    logger.info(f"history: {history}")

    history[-1][1] = ""
    ownHistory = []
    if history_size>0:
        ownHistory = history[-history_size:-1]

    messages = []
    for user, assistant in ownHistory:
        messages.append({
            "role": "user",
            "content": user
        })
        messages.append({
            "role": "assistant",
            "content": assistant
        })
    messages.append({
        "role": "user",
        "content": message
    })

    logger.info(f"请求大模型内容messages：{messages}")

    answerStr = ""
    for answer in model.generate_response(messages, enable_thinking=enable_thinking, max_new_tokens=32768):
        answer = answer.replace("<think>", "")
        history[-1][1] += answer
        answerStr += answer
        yield history

    logger.info(f"用户ID：{user_id}，大模型请求总耗时（秒）：{time.time() - start_time}，结果：{answerStr}")

############################################################################################################################

def user(user_message, history):
    return "", history + [[user_message, None]]

with gr.Blocks() as demo:
    with gr.Row():
        history_size = gr.Number(value=0, label="上下文对话轮数(0为不使用上下文)", scale=4)
        enable_thinking = gr.Checkbox(value=False, label="开启思考", info="*", scale=4)
    chatbot = gr.Chatbot(label='Qwen3-0.6B', height=700)
    msg = gr.Textbox(label='请在这里输入你的问题', value='中国首都在哪里？')

    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        chat, [history_size, enable_thinking, chatbot], chatbot
    )

def main():
    # 启动Gradio应用
    demo.queue(status_update_rate="auto", default_concurrency_limit=120, max_size=120)
    demo.launch(server_name='0.0.0.0', server_port=7860, share=True, root_path='/ai',)


if __name__ == "__main__":
    main()

