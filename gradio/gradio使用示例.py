import re

# https://github.com/gradio-app/gradio.git

#########################################################################################################################
import gradio as gr

def greet(name):
    return "Hello " + name + "!"

demo = gr.Interface(fn=greet,# fn : 被UI包装的函数
                    # inputs="text", # inputs : 作为输入的组件 (例如： "text", "image" or "audio")
                    inputs=gr.Textbox(lines=2, placeholder="今天天气怎么样？"), # 输入框也可以这样控制行数，及提示语等：
                    outputs="text", # outputs : 作为输出的组件 (例如： "text", "image" or "label")
                    )
demo.launch(share=True)

# 浏览器打开  http://127.0.0.1:7860 即可访问问答界面；

#########################################################################################################################
# 添加一个登录认证
import gradio as gr

def greet(name):
    return "Hello " + name + "!"

demo = gr.Interface(fn=greet, inputs="text", outputs="text", concurrency_limit=10)

demo.launch(server_name='0.0.0.0',
         server_port=8082,
         show_api=False,
         share=True,
         inbrowser=False,
         auth=("zhangsan", '123456'),
         )

#########################################################################################################################
# 多输入，多输出：
import gradio as gr

def greet(name, is_um: bool, temperature: int):
    '''
    该函数接受字符串、布尔值和数字，并返回字符串和数字。观察应该如何传递输入和输出组件列表。
    :param name:
    :param is_um: 是否是um
    :param temperature: 温度
    :return:
    '''
    salutation = "学生" if is_um else "老师"
    greeting = f"{name} {salutation}. 今天 {temperature} ℃"
    celsius = (temperature - 32) * 5 / 9
    return greeting, round(celsius, 2)

demo = gr.Interface(
    fn=greet,
    inputs=[gr.Textbox(label="姓名"), gr.Checkbox(label="是否学生"), gr.Slider(0, 100, label="温度")],
    outputs=["text", "number"],
)
demo.launch(share=True)

#########################################################################################################################
# 输入、输出均为图像：
# 您只需将组件包装在列表中。输入列表inputs中的每个组件依次对应函数的一个参数。输出列表outputs中的每个组件都对应于函数的一个返回值，两者均按顺序对应。
#
# 一个图像示例
# Gradio支持多种类型的组件，如 Image、DateFrame、Video或Label 。让我们尝试一个图像到图像的函数来感受一下！

import numpy as np
import gradio as gr

def sepia(input_img):
    sepia_filter = np.array([
        [0.393, 0.769, 0.189],
        [0.349, 0.686, 0.168],
        [0.272, 0.534, 0.131]
    ])
    sepia_img = input_img.dot(sepia_filter.T)
    sepia_img /= sepia_img.max()
    return sepia_img

demo = gr.Interface(sepia, gr.Image(), "image")
# 可以用 type= 关键字参数设置组件使用的数据类型。例如，如果你想让你的函数获取一个图像的文件路径，而不是一个NumPy数组时，输入 Image 组件可以写成：
# gr.Image(type="filepath")
# 还要注意，我们的输入 Image 组件带有一个编辑按钮 🖉，它允许裁剪和放大图像。

demo.launch(share=True)

#########################################################################################################################
# 与Interface相比，Blocks: 更加灵活且可控
import gradio as gr

def greet(name):
    return "你好 " + name + "!"

with gr.Blocks() as demo:
    name = gr.Textbox(label="姓名")
    output = gr.Textbox(label="输出结果")
    greet_btn = gr.Button("运行")
    greet_btn.click(fn=greet, inputs=name, outputs=output)
# Blocks 由 with 子句组成，在该子句中创建的任何组件都会自动添加到应用程序中。
# 组件在应用程序中按创建的顺序垂直显示
# 一个 按钮 Button 被创建，然后添加了一个 click 事件监听器。这个API看起来很熟悉！就像 Interface一样， click 方法接受一个Python函数、输入组件和输出组件。
demo.launch(share=True)

# Blocks布局：Blocks是Gradio中用于自定义布局的一种强大工具，允许用户以更灵活的方式组织界面元素。
# 布局组件：Row, Column, Tab, Group, Accordion
# Row和Column：分别用于创建水平行和垂直列的布局。
# * 示例：使用Row来水平排列几个按钮，使用Column来垂直排列一系列输入组件。
# Tab：用于在TabbedInterface中创建各个标签页。
# * 示例：在一个应用中创建多个Tab，每个标签页包含特定主题的内容。
# Group：将多个组件组合成一个组，便于统一管理和布局。
# * 示例：创建一个包含多个相关输入组件的Group。
# Accordion：创建可以展开和折叠的面板，用于管理空间和改善界面的可用性。
# * 示例：将不常用的选项放入Accordion中，以减少界面的拥挤。

#########################################################################################################################
# 更为复杂的自定义布局
import numpy as np
import gradio as gr

def flip_text(x):
    return x[::-1]

def flip_image(x):
    return np.fliplr(x)

with gr.Blocks() as demo:
    gr.Markdown("演示文本和图像切换.")
    with gr.Tabs():
        with gr.TabItem("文本框"):
            text_input = gr.Textbox(placeholder="今天天气怎么样？", label="输入文本内容")
            text_output = gr.Textbox(placeholder="天气不错!", label="文本运行结果")
            text_button = gr.Button("运行文本")
        with gr.TabItem("图像框"):
            with gr.Row():
                image_input = gr.Image(label="上传图像")
                image_output = gr.Image(label="输出图像")
            image_button = gr.Button("运行图像")

    text_button.click(flip_text, inputs=text_input, outputs=text_output)
    image_button.click(flip_image, inputs=image_input, outputs=image_output)

demo.launch()

#########################################################################################################################
# 流式输出
import gradio as gr
from pydub import AudioSegment
from time import sleep

with gr.Blocks() as demo:
    input_audio = gr.Audio(label="Input Audio", type="filepath", format="mp3")
    with gr.Row():
        with gr.Column():
            stream_as_file_btn = gr.Button("Stream as File")
            format = gr.Radio(["wav", "mp3"], value="wav", label="Format")
            stream_as_file_output = gr.Audio(streaming=True)

            def stream_file(audio_file, format):
                audio = AudioSegment.from_file(audio_file)
                i = 0
                chunk_size = 1000
                while chunk_size * i < len(audio):
                    chunk = audio[chunk_size * i : chunk_size * (i + 1)]
                    i += 1
                    if chunk:
                        file = f"/tmp/{i}.{format}"
                        chunk.export(file, format=format)
                        yield file
                        sleep(0.5)

            stream_as_file_btn.click(
                stream_file, [input_audio, format], stream_as_file_output
            )

            gr.Examples(
                [["audio/cantina.wav", "wav"], ["audio/cantina.wav", "mp3"]],
                [input_audio, format],
                fn=stream_file,
                outputs=stream_as_file_output,
            )

        with gr.Column():
            stream_as_bytes_btn = gr.Button("Stream as Bytes")
            stream_as_bytes_output = gr.Audio(format="bytes", streaming=True)

            def stream_bytes(audio_file):
                chunk_size = 20_000
                with open(audio_file, "rb") as f:
                    while True:
                        chunk = f.read(chunk_size)
                        if chunk:
                            yield chunk
                            sleep(1)
                        else:
                            break
            stream_as_bytes_btn.click(stream_bytes, input_audio, stream_as_bytes_output)

demo.launch(share=True)

######################################################################################################################
# 显示进度条
import gradio as gr
import time
# from https://gradio.app/docs/#progress
def my_function(x=10, progress_demo=gr.Progress()):

    x=int(x)

    gr.Info("Starting process", visible=False)
    if 0 < x < 10:
        gr.Warning("设置太小！") # 页面中弹出警告
    if x <= 0:
        raise gr.Error("应该设置大于0！")  # 页面弹出错误

    progress_demo(0, desc="开始...") # 将进度条初始化为0，并显示描述
    time.sleep(1)
    for i in progress_demo.tqdm(range(x), desc="完成"):
        time.sleep(0.1)
    res=f'run {x} steps'
    return res

# 页面“提交”等按钮设置为中文
demo = gr.Interface(my_function,
            gr.Number(label="输入"),
            gr.Textbox(label="输出"),
            title='gradio模型预测',
            description="这是一个演示demo",
            article="提示：AI自动生成，仅供参考",
            theme=gr.themes.Soft(), # 通过 theme 参数来设置应用程序的主题
            submit_btn = gr.Button("提交"),
            stop_btn = "停止",
            clear_btn = "清除",
            flagging_options = [("标记", "")]
                    )

demo.launch(share=True)

######################################################################################################################
# 数据标记
import gradio as gr
import random
import typing

def greet(name):
    if len(name) == 0:
        return 0
    elif len(name) < 5:
        return 1
    else:
        return 2

demo = gr.Interface(fn=greet,# fn : 被UI包装的函数
                    inputs=gr.Textbox(lines=1, placeholder="今天天气怎么样？"), # 输入框也可以这样控制行数，及提示语等：
                    outputs=gr.Label(label="输出"), # outputs : 作为输出的组件 (例如： "text", "image" or "label")
                    title='gradio模型预测', # 接受文本格式，可以在接口的最顶部显示它，同时它也成为了页面标题。
                    description="这是一个演示demo", # 接受文本、Markdown 或 HTML，并将其放置在标题下方。
                    article="提示：AI自动生成，仅供参考", # 接受文本、Markdown 或 HTML，并将其放置在接口下方。
                    theme=gr.themes.Soft(),  # 通过 theme 参数来设置应用程序的主题
                    submit_btn=gr.Button("提交", elem_id="custom-button"),
                    stop_btn="停止",
                    clear_btn="清除",
                    flagging_options=[("对", "1"), ("错", "0")], # 元组列表：每个元组的形式为 (label, value)，其中 label 是显示在按钮上的字符串，value 是存储在标记 CSV 中的字符串。
                    flagging_dir="./tmp2", # 存储标记数据的目录的名称
                    )
# 自定义 CSS, 设置按钮颜色等
demo.css = """
    #custom-button {
        background-color: #4CAF50; /* Green */
        border: none;
        color: white;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        padding: 10px 24px;
    }
"""
demo.launch(share=True)

###########################################################################################################################
# 聊天界面：ChatInterface允许创建类似聊天应用的界面，适用于构建交云式聊天机器人或其他基于文本的交互式应用。
import gradio as gr
import re

def login(username, password):
    if re.search('\d', username):
        raise gr.Error("登录错误")
    return True

def slow_echo(message, history, request: gr.Request):
    if request:
        print("Request headers dictionary:", dict(request.headers))
        print("Query parameters:", dict(request.query_params))
        print("IP address:", request.client.host)
        print("Gradio session hash:", request.session_hash)
        print("username:", request.username)

    history = history[-3:]  # 获取最近的三条历史对话；
    gr.Info(f"历史消息:{history}")

    return f"机器人回复: {message} "

demo = gr.ChatInterface(slow_echo)
demo.launch(share=True, auth=login, auth_message="欢迎登录987", debug=True)
# launch方法中提供了一个参数auth，可以方便地增加用户登录功能。

###########################################################################################################################
# 文本分类的演示系统
import gradio as gr
# 导入transformers相关包
from transformers import pipeline
# 通过Interface加载pipeline并启动服务
gr.Interface.from_pipeline(pipeline("text-classification", model="uer/roberta-base-finetuned-dianping-chinese")).launch()
# gr.Interface.from_pipeline(pipeline("text-generation", model="qwen/Qwen2-1.5B")).launch()


# 输入组件 (Inputs)
# Audio：允许用户上传音频文件或直接录音。参数：source: 指定音频来源（如麦克风）、type: 指定返回类型。 示例：gr.Audio(source="microphone", type="filepath")
# Checkbox：提供复选框，用于布尔值输入。参数：label: 显示在复选框旁边的文本标签。 示例：gr.Checkbox(label="同意条款")
# CheckboxGroup：允许用户从一组选项中选择多个。参数：choices: 字符串数组，表示复选框的选项、label: 标签文本。示例：gr.CheckboxGroup(["选项1", "选项2", "选项3"], label="选择你的兴趣")
# ColorPicker：用于选择颜色，通常返回十六进制颜色代码。参数：default: 默认颜色值。示例：gr.ColorPicker(default="#ff0000")
# Dataframe：允许用户上传CSV文件或输入DataFrame。参数：headers: 列标题数组、row_count: 初始显示的行数。示例：gr.Dataframe(headers=["列1", "列2"], row_count=5)
# Dropdown：下拉菜单，用户可以从中选择一个选项。参数：choices: 字符串数组，表示下拉菜单的选项、label: 标签文本。示例：gr.Dropdown(["选项1", "选项2", "选项3"], label="选择一个选项")
# File：用于上传任意文件，支持多种文件格式。参数：file_count: 允许上传的文件数量，如"single"或"multiple"、type: 返回的数据类型，如"file"或"auto"。示例：gr.File(file_count="single", type="file")
# Image：用于上传图片，支持多种图像格式。参数：type图像类型，如pil。示例：gr.Image(type='pil')
# Number：数字输入框，适用于整数和浮点数。参数：default: 默认数字、label: 标签文本。示例：gr.Number(default=0, label="输入一个数字")
# Radio：单选按钮组，用户从中选择一个选项。参数：choices: 字符串数组，表示单选按钮的选项、label: 标签文本。示例：gr.Radio(["选项1", "选项2", "选项3"], label="选择一个选项")
# Slider：滑动条，用于选择一定范围内的数值。参数：minimum: 最小值、maximum: 最大值、step: 步长、label: 标签文本。示例：gr.Slider(minimum=0, maximum=10, step=1, label="调整数值")
# Textbox：单行文本输入框，适用于简短文本。参数：default: 默认文本、placeholder: 占位符文本。示例：gr.Textbox(default="默认文本", placeholder="输入文本")
# Textarea：多行文本输入区域，适合较长的文本输入。参数：lines: 显示行数、placeholder: 占位符文本。示例：gr.Textarea(lines=4, placeholder="输入长文本")
# Time：用于输入时间。参数：label: 标签文本。示例：gr.Time(label="选择时间")
# Video：视频上传组件，支持多种视频格式。参数：label: 标签文本。示例：gr.Video(label="上传视频")
# Data：用于上传二进制数据，例如图像或音频的原始字节。参数：type: 数据类型，如"auto"自动推断。示例：gr.Data(type="auto", label="上传数据")


# 输出组件 (Outputs)
# Audio：播放音频文件。参数：type 指定输出格式。示例：gr.Audio(type="auto")
# Carousel：以轮播方式展示多个输出，适用于图像集或多个数据点。参数：item_type 设置轮播项目类型。示例：gr.Carousel(item_type="image")
# Dataframe：展示Pandas DataFrame，适用于表格数据。参数：type 指定返回的DataFrame类型。示例：gr.Dataframe(type="pandas")
# Gallery：以画廊形式展示一系列图像。
# HTML：展示HTML内容，适用于富文本或网页布局。
# Image：展示图像。参数：type 指定图像格式。 示例：gr.Image(type="pil")
# JSON：以JSON格式展示数据，便于查看结构化数据。
# KeyValues：以键值对形式展示数据。
# Label：展示文本标签，适用于简单的文本输出。
# Markdown：支持Markdown格式的文本展示。
# Plot：展示图表，如matplotlib生成的图表。
# Text：用于显示文本，适合较长的输出。
# Video：播放视频文件。


