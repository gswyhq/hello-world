#!/usr/bin/env python
# coding=utf-8

import os
import gradio as gr
from transformers import AutoModelWithLMHead,AutoTokenizer,pipeline
USERNAME = os.getenv("USERNAME")

# 模型来源：https://www.modelscope.cn/models/cubeai/trans-opus-mt-en-zh/files
mode_name = rf"D:\Users\{USERNAME}\data\trans-opus-mt-en-zh"
model = AutoModelWithLMHead.from_pretrained(mode_name)
tokenizer = AutoTokenizer.from_pretrained(mode_name)
translation = pipeline("translation_en_to_zh", model=model, tokenizer=tokenizer)
result = translation('I like to study Data Science and Machine Learning.', max_length=400)
print(result[0]['translation_text'])

def TranslationModel(en, *args): # en: 名字可任取，inputs传入的值 会传给en
    result = translation(en, max_length=400)
    return result[0]['translation_text']

en2ch = gr.Interface(
    fn=TranslationModel,
    inputs=[
        gr.Textbox(label="请输入待翻译的文本"),
        gr.Dropdown(label="源语言", choices=["英文"]),
        gr.Dropdown(label="目标语言", choices=['简体中文']),
    ],
    submit_btn=gr.Button("翻译", elem_id='centered-button'),
    clear_btn=None, flagging_options=[],
    outputs=gr.Textbox(label="翻译结果"),
    # example_labels=["示例"],
    examples=[["Building a translation demo with Gradio is so easy!", "英文", "简体中文"]],
    cache_examples=False,
    title="翻译演示系统",
    description="模型来源于：[Helsinki-NLP](https://hf-mirror.com/Helsinki-NLP/opus-mt-en-zh) "
)

en2ch.css = '''#centered-button {
    width: 100%;
    border: 2px solid gray;
    border-radius: 12px;
    padding: 10px;
    background-color: green;
    cursor: pointer;
    display: flex;
    justify-content: center;
    align-items: center;
    box-sizing: border-box;
}'''
def main():
    en2ch.launch(server_name='0.0.0.0',
         server_port=8082, share=True)


if __name__ == "__main__":
    main()
