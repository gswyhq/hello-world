#!/usr/bin/env python
# coding=utf-8

# 使用streamer参数实现流式输出

import time
from threading import Thread
from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import torch
import os

import platform

USERNAME = os.getenv("USERNAME")
# 设置环境变量
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

if platform.system() == 'Windows':
    DEEPSEEK_MODEL_PATH = rf"D:\Users\{USERNAME}\data\DeepSeek-R1-Distill-Qwen-1.5B"
else:
    DEEPSEEK_MODEL_PATH = "/root/DeepSeek-R1-Distill-Qwen-1.5B"


class DeepSeekModel:
    def __init__(self, model_path="/root/DeepSeek-R1-Distill-Qwen-1.5B"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # 加载tokenizer和模型
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)

        # 设置模型为评估模式
        self.model.eval()

    def generate_response(self, prompt, max_new_tokens=2048, temperature=0.6, top_p=0.7, top_k=16, repetition_penalty=1.0):
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
            generation_kwargs = dict(input_ids=inputs.input_ids, streamer=streamer, max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p, top_k=top_k,
                        repetition_penalty=repetition_penalty,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id)
            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()
            generated_text = ""
            for text in streamer:
                generated_text += text
                yield text
            print("模型输出结果：", generated_text)
        except Exception as e:
            print(f"生成回答时发生错误: {str(e)}")
            return None


# 初始化模型
model = DeepSeekModel(model_path=DEEPSEEK_MODEL_PATH)

# for answer in model.generate_response('中国首都是哪里？<think>', max_new_tokens=20, temperature=0.6, top_p=0.7, top_k=16, repetition_penalty=1.0):
#     print(answer)


############################################################################################################################
import gradio as gr


def chat(history_size, history, request: gr.Request):
    user_id = request.session_hash
    message = history[-1][0]
    # start_time = time.time()
    logger.info(f"用户ID：{user_id}，请求的问题：{message}, 保留对话轮数：{history_size}")
    history[-1][1] = ""
    ownHistory = history[-history_size:-1] if history_size > 0 else []

    for answer in model.generate_response(message, max_new_tokens=20000):
        history[-1][1] += answer
        yield history


############################################################################################################################

def user(user_message, history):
    return "", history + [[user_message, None]]

with gr.Blocks() as demo:

    history_size = gr.Number(value=0, label="上下文对话轮数(0为不使用上下文)", step=1, precision=0)
    chatbot = gr.Chatbot(label='DeepSeek-R1-Distill-Qwen-1.5B模型问答对话框', height=700)
    msg = gr.Textbox(label='请在这里输入你的问题', value='中国首都在哪里？')

    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        chat, [history_size, chatbot], chatbot
    )


def main():
    # 启动Gradio应用
    demo.queue(status_update_rate="auto", default_concurrency_limit=120, max_size=120)
    demo.launch(server_name='0.0.0.0', server_port=7860, share=True,
                # auth=('admin', '123456')
                )



if __name__ == "__main__":
    main()



