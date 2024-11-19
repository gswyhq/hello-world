#!/usr/bin/env python
# coding=utf-8

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
os.environ['PATH'] = f'D:\\Users\\{USERNAME}\\ffmpeg-7.1\\bin\\;' + os.getenv('PATH')
print(os.getenv('PATH'))  # 输出新的PATH环境变量

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# model_id = "openai/whisper-large-v3-turbo"
model_id = rf"D:\Users\{USERNAME}\data\whisper-large-v3-turbo"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)

# dataset_path = "distil-whisper/librispeech_long"
dataset_path = rf"D:\Users\{USERNAME}\data\whisper-large-v3-turbo/librispeech_long"
# dataset = load_dataset("parquet", data_files=f"{dataset_path}\clean/validation-00000-of-00001-913508124a40cb97.parquet", split="train")
# sample = dataset[0]["audio"]

# translate: 将音频翻译为英文（目前只能是翻译成英文）；language: 源音频文件语言；
# generate_kwargs={"language": "chinese", "task": "translate"}
# result = pipe(sample, generate_kwargs={"language": "chinese", "task": "translate"})
# result = pipe(rf"D:\Users\{USERNAME}\Downloads\1726售后非车机器人-非车有意向工单.mp3")
# result = pipe(rf"D:\Users\{USERNAME}\Downloads\城市保AI外呼.wav")
# print("音频转文本的结果：", result["text"])

# 将本地音频文件转录只需在调用管道时传递音频文件的路径：
# result = pipe("audio.mp3")

# 多个音频文件可以通过指定它们为列表并设置 batch_size 参数来并行转录：
# result = pipe(["audio_1.mp3", "audio_2.mp3"], batch_size=2)

def audio2text(file_name, file_name2, format):
    print('上传的音频文件：', file_name)
    print('上传的视频文件：', file_name2)
    result = pipe(file_name or file_name2)
    print("音视频转文本的结果：", result["text"])
    return result['text']

##########################################################################################################################

with gr.Blocks() as demo:
    with gr.TabItem("选择上传文件(选择其一即可)"):
        input_audio = gr.Audio(label="Input Audio", type="filepath", format=None)
        input_video = gr.Video(label="输入视频", format=None)
    stream_as_file_btn = gr.Button("开始转录")
    format = gr.Radio(["wav", "mp3", "mp4"], value="wav", label="Format")
    output = gr.Textbox(label="输出结果")
    stream_as_file_btn.click(
        fn=audio2text, inputs=[input_audio, input_video, format], outputs=output
    )

def main():
    demo.launch(share=True)


if __name__ == "__main__":
    main()
