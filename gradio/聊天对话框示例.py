#!/usr/bin/env python
# coding=utf-8

import os
import pandas as pd
import json

USERNAME = os.getenv("USERNAME")

###########################################################################################################################
# 使用ChatInterface搭建聊天界面
import gradio as gr

def chat(message, history):
	"""
	用户输入后的回调函数 random_response
	参数：
	message: 用户此次输入的消息
	history: 历史聊天记录，比如 [["use input 1", "assistant output 1"],["user input 2", "assistant output 2"]]
	返回值：输出的内容
	"""
	return f"你好：{message}"


demo = gr.ChatInterface(chat)

###########################################################################################################################
# 使用ChatInterface搭建稍微复杂点的聊天界面
def yes_man(message, history):
	if message.endswith("?"):
		return "Yes"
	else:
		return "Ask me anything!"

demo = gr.ChatInterface(
	yes_man,
	# type="messages",
	chatbot=gr.Chatbot(height=300,placeholder="<strong>针对表格内容</strong><br>您可以询问任何问题"),
	textbox=gr.Textbox(placeholder="请在此输入您的问题", container=False, scale=7),
	title="基于表格的问答系统",
	description="先选择一个数据表格，再基于表格进行提问",
	theme="soft",
	# examples=[{'text': '长江流域的小型水库的库容总量是多少？'}, {'text': '那平均值是多少？'}, {'text': '那水库的名称呢？'}, {'text': '换成中型的呢？'}],
	examples=['长江流域的小型水库的库容总量是多少？', '那平均值是多少？', '那水库的名称呢？', '换成中型的呢？'],
)

###########################################################################################################################
# 使用用户的输入和历史记录
import random  # 导入random模块
import gradio as gr  # 导入gradio库，并简称为gr


# 定义一个根据历史长度奇偶性交替同意或不同意的函数
def alternatingly_agree(message, history):
	# 如果历史记录的长度是偶数，则同意
	if len(history) % 2 == 0:
		return f"同意： '{message}'"  # 同意用户的观点
	# 否则，表示不同意
	else:
		return "不同意"  # 不同意用户的观点


# 使用Gradio的ChatInterface模块创建一个聊天界面，使用上面定义的alternatingly_agree函数来响应用户的输入
demo = gr.ChatInterface(alternatingly_agree)

###########################################################################################################################
# 流媒体聊天机器人
# 在您的聊天功能中，您可以使用 yield 生成一系列部分响应，每个响应都会替换前一个。

import time  # 导入time模块，用于实现等待（延时）
import gradio as gr  # 导入gradio库，并简称为gr


# 定义一个慢速回显函数，模拟逐字打印的效果
def slow_echo(message, history):
	# 遍历消息中的每一个字符
	for i in range(len(message)):
		time.sleep(0.1)  # 每输出一个字符后暂停0.1秒
		yield "You typed: " + message[: i + 1]  # 逐渐展示已经输入的消息内容；通过这种方式，可以使得消息像是被逐字被打印出来；


# 使用Gradio的ChatInterface模块创建一个聊天界面，使用上面定义的slow_echo函数来响应用户的输入
demo = gr.ChatInterface(slow_echo)

############################################################################################################################
# 定制您的聊天机器人
import gradio as gr  # 导入gradio库


# 定义一个总是回答"Yes"或提示用户提问的函数
def yes_man(message, history):
	# 如果消息以问号结尾，则回答"Yes"
	if message.endswith("?"):
		return "Yes"
	else:
		return f"你好：{message}"


# 使用Gradio的ChatInterface模块创建一个聊天界面
demo = gr.ChatInterface(
	yes_man,  # 指定yes_man函数为回答逻辑
	chatbot=gr.Chatbot(height=300),  # 创建一个聊天机器人并设置其容器高度为300
	textbox=gr.Textbox(placeholder="<strong>Your Personal Yes-Man</strong><br>Ask Me Anything",
					   container=False, scale=7),
	# 创建文本输入框并设置提示文本、不使用容器模式，并放大7倍
	title="Yes Man",  # 设置聊天界面标题
	description="Ask Yes Man any question",  # 设置聊天界面描述
	theme="soft",  # 设置界面主题为soft
	examples=["Hello", "Am I cool?", "Are tomatoes vegetables?"],  # 提供示例问题
	cache_examples=True,  # 启用缓存示例功能，若设置为true,点击示例后，则在页面上回答该问题的答案；若设置为false(默认)，则将该示例输入对话框，但还需人工点击回车或发送才可以有答案；
)

###########################################################################################################################
# 输入’重置‘清空历史对话消息，history=[]
import gradio as gr
import random
import time


def respond(message, chat_history):
    print("chat_history", chat_history)
    bot_message = f"你好：{message}"
    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": bot_message})
    if message in ['重置', '清空消息']:
        chat_history = []
    return "", chat_history

with gr.Blocks() as demo:
    chatbot = gr.Chatbot(type="messages", label='问答对话框')
    msg = gr.Textbox(label='请在这里输入你的问题', value='今天天气怎么样？')

    msg.submit(respond, [msg, chatbot], [msg, chatbot])

demo.launch(server_name='0.0.0.0', server_port=7860, share=True)

###########################################################################################################################
# 输入’重置‘清空历史对话消息，history=[],并输出多个数据窗口，包括除对话窗口外的其他窗口

import gradio as gr

def chat(message, history):
    # 创建一个新的空历史记录，以避免影响后续的调用
    print("history=", history)

    history.append({"role": "user", "content": message})
    if message in ['重置', '清空消息']:
        history = []

    if "python" in message.lower():
        bot_message = "Type Python or JavaScript to see the code."
        history.append({"role": "assistant", "content": bot_message})
        return "", gr.Code(language="python", value="python_code"), history
    elif "javascript" in message.lower():
        bot_message = "Type Python or JavaScript to see the code."
        history.append({"role": "assistant", "content": bot_message})
        return "", gr.Code(language="javascript", value="js_code"), history
    else:
        bot_message = "Please ask about Python or JavaScript."
        history.append({"role": "assistant", "content": bot_message})
        return "", None, history

with gr.Blocks() as demo:
    code = gr.Code(render=False)
    with gr.Row():
        with gr.Column():
            gr.Markdown("<center><h1>Write Python or JavaScript</h1></center>")
            chatbot = gr.Chatbot(type="messages", label='问答对话框')
            msg = gr.Textbox(label='请在这里输入你的问题', value='今天天气怎么样？')
        with gr.Column():
            gr.Markdown("<center><h1>Code Artifacts</h1></center>")
            code.render()

    msg.submit(chat, [msg, chatbot], [msg, code, chatbot])

demo.launch(server_name='0.0.0.0', server_port=7860, share=True)

###########################################################################################################################
# 多模态功能添加到您的聊天机器人
# 您可能希望为您的聊天机器人添加多模态功能。
# 例如，您可能希望用户能够轻松地上传图片或文件到您的聊天机器人并对其提问。您可以通过向 gr.ChatInterface 类传递一个参数(multimodal=True)来使您的聊天机器人"多模态"。

import gradio as gr  # 导入gradio库，用于创建交互式界面
import time  # 导入time模块，虽然在这个例子中未直接使用，但可用于其他功能如延迟


# 定义一个函数，用于计算用户上传的文件数量
def count_files(message, history):
	'''当 multimodal=True 时， fn 的签名会略有变化。
	你的函数的第一个参数应该接受一个由提交的文本和上传的文件组成的字典，看起来像这样： {"text": "user input", "file": ["file_path1", "file_path2", ...]}
	'''
	num_files = len(message["files"])  # 计算上传文件的数量

	for file_name in message['files']:
		if file_name.lower().endswith("xlsx"):
			df = pd.read_excel(file_name)
			return gr.HTML(df.to_html(index=False))
		if file_name.lower().endswith("wav"):
			return gr.Audio(file_name)
		if file_name.lower().endswith(".jpg"):
			return gr.Image(file_name)
	# 返回复杂的Responses, 但貌似不支持多种输出数据组合；
	# 上面介绍的还都是返回字符串，称之为chatbot中的text，也可以返回更加复杂的结构：
	# gr.Image
	# gr.Plot
	# gr.Audio
	# gr.HTML
	# gr.Video
	# gr.Gallery
	return f"你好：{message['text']}\n你上传了 {num_files} 个文件，分别为：{'、'.join(message['files'])}"  # 返回一个字符串，包含上传的文件数量


# 创建一个Gradio聊天界面
demo = gr.ChatInterface(
	fn=count_files,  # 指定处理消息的函数是count_files
	# textbox=gr.MultimodalTextbox(), # 如果要自定义多模态聊天机器人的文本框的 UI/UX，您应该将 gr.MultimodalTextbox 的实例传递给 ChatInterface 的 textbox 参数，而不是 gr.Textbox 的实例
	examples=[{"text": "Hello", "files": []}],  # 提供一个示例输入，没有附带文件
	title="基于上传文档的问答",  # 设置界面的标题为"Echo Bot"
	multimodal=True,  # 开启多模态支持，允许文本和文件同时作为输入
	cache_examples=True,
)


###########################################################################################################################
# 问答对话框，并额外展示生成的代码
import gradio as gr

python_code = """
def fib(n):
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fib(n-1) + fib(n-2)
"""

js_code = """
function fib(n) {
    if (n <= 0) return 0;
    if (n === 1) return 1;
    return fib(n - 1) + fib(n - 2);
}
"""

def chat(message, history):
    if "python" in message.lower():
        return "Type Python or JavaScript to see the code.", gr.Code(language="python", value=python_code)
    elif "javascript" in message.lower():
        return "Type Python or JavaScript to see the code.", gr.Code(language="javascript", value=js_code)
    else:
        return f"请输入 Python 或 JavaScript.{gr.__version__}", None

with gr.Blocks() as demo:
    code = gr.Code(render=False)
    with gr.Row():
        with gr.Column():
            gr.Markdown("<center><h1>生成 Python 或 JavaScript代码</h1></center>")
            gr.ChatInterface(
                chat,
                examples=["Python", "JavaScript"],
                additional_outputs=[code],
                type="messages"
            )
        with gr.Column():
            gr.Markdown("<center><h1>生成的代码</h1></center>")
            code.render()

###########################################################################################################################

# 添加额外的输入组件。在输入问题时候，输入最大输出字符数

import gradio as gr  # 导入gradio库，用于创建交云式界面
import time  # 导入time模块，用于在循环中添加延迟效果


# 定义一个echo函数，模拟逐字输出消息的效果
def echo(message, history, system_prompt, tokens):
	# 格式化字符串，包含系统提示和用户消息
	response = f"最大输出字符数为：{tokens}, {type(tokens)}\n系统提示: {system_prompt}\n 消息: {message}."
	# 通过最小值函数确定输出字符的最大长度，避免超出用户设置的token数
	for i in range(min(len(response), int(tokens))):
		time.sleep(0.05)  # 在每个字符输出之间添加短暂延迟，增加逐字显示效果
		yield response[: i + 1]  # 逐步输出字符串，达到"打字机"效果


# 使用Gradio的ChatInterface构建一个支持多输入的聊天界面
demo = gr.ChatInterface(
	echo,  # 将echo函数设为回调函数
	# 通过additional_inputs参数添加额外的输入控件
	additional_inputs=[
		gr.Textbox("你是一个全能的AI.", label="系统提示"),  # 添加一个文本输入框，用于输入系统提示
		gr.Slider(10, 100),  # 添加一个滑动条，范围从10到100，用于控制输出的最大字符数（tokens）
	],
	additional_inputs_accordion="请选择输出的最大字符数"
).queue()

##########################################################################################################################
# 实现一个选择读取展示excel表的页面，选择下拉列表，展示表格内容
import gradio as gr
import pandas as pd

down_lst= ['fund', 'car', 'student', 'reservoir', 'car_sales']
with gr.Blocks() as demo:
	# 设置一个下拉框用来选择excel文件
	num=gr.Dropdown(down_lst,label="选择数据库")
	# 设置一个按钮 更新表格内容
	lion_button = gr.Button("查看数据库内容")
	# 设置一个Dataframe 表格用来展示excel表的内容
	excel_df=gr.Dataframe(headers=None)

	#写一个调用函数，当按下"更新表格"的按钮后，读取对应文件，并更新表格内容
	def read_excel(table_id):
		df1= pd.read_excel(rf"D:\Users\{USERNAME}\Downloads\test4\{table_id}.xlsx")
		return {
			excel_df:df1
		}

	lion_button.click(
		read_excel,
		num,
		excel_df
	)
##########################################################################################################################
# 添加额外的输入组件。在输入问题时候，选择数据库
def text2sql(message, history, table_id):
	return f"你好：{message}, \n选择数据库是table_id: {table_id}"


with gr.Blocks() as demo:
	gr.ChatInterface(
		text2sql,
		# type="messages",
		chatbot=gr.Chatbot(height=300, placeholder="<strong>针对表格内容</strong><br>您可以询问任何问题"),
		textbox=gr.Textbox(placeholder="请在此输入您的问题", container=False, scale=7, label='问题'),
		title="基于表格的问答系统",
		description="先选择一个数据表格，再基于表格进行提问",
		theme="soft",
		# examples=[{'text': '长江流域的小型水库的库容总量是多少？'}, {'text': '那平均值是多少？'}, {'text': '那水库的名称呢？'}, {'text': '换成中型的呢？'}],
		examples=[['长江流域的小型水库的库容总量是多少？', 'reservoir'],
				  ['那平均值是多少？', 'reservoir'],
				  ['那水库的名称呢？', 'reservoir'],
				  ['换成中型的呢？', 'reservoir']],
		cache_examples=True,
		# 通过additional_inputs参数添加额外的输入控件,也可以是一个参数值
		additional_inputs=[
			gr.Dropdown(choices=['fund', 'car', 'student', 'reservoir', 'car_sales'], value='reservoir', multiselect = False, allow_custom_value= False, label='数据库')
		],
		additional_inputs_accordion="请选择问答数据库"
	)

##########################################################################################################################
# 添加额外的输入组件。在输入问题时候，选择数据库，并展示数据库内容，再输入问题进行问答；

TABLE_NAME2ID_MAP = {'基金': 'fund',
 '汽车': 'car',
 '学生信息': 'student',
 '水库': 'reservoir',
 '汽车销量表': 'car_sales'}

def text2sql(message, history, table_name):
	return f"你好：{message}, \n选择数据库是table_name: {table_name}"



with gr.Blocks() as demo:
	# 设置一个下拉框用来选择excel文件
	default_table_name = '水库'
	default_table_id = 'reservoir'
	table_name = gr.Dropdown(list(TABLE_NAME2ID_MAP.keys()),label="选择数据库", value=default_table_name)
	# 设置一个按钮 更新表格内容
	lion_button = gr.Button("查看数据库内容")

	# 设置一个Dataframe 表格用来展示excel表的内容
	excel_df=gr.Dataframe(pd.read_excel(rf"D:\Users\{USERNAME}\Downloads\test4\{default_table_id}.xlsx"))

	#写一个调用函数，当按下的按钮后，读取对应文件，并更新表格内容
	def read_excel(table_name):
		table_id = TABLE_NAME2ID_MAP[table_name]
		df1= pd.read_excel(rf"D:\Users\{USERNAME}\Downloads\test4\{table_id}.xlsx")
		# 应用样式
		df1 = df1.style.highlight_max(color='lightgreen', axis=0)
		return {
			excel_df:df1
		}

	lion_button.click(
		read_excel,
		table_name,
		excel_df
	)
	gr.ChatInterface(
		text2sql,
		# type="messages",
		title="基于表格的问答系统",
		description="先选择一个数据表格，再基于表格进行提问",
		chatbot=gr.Chatbot(height=300,placeholder="<strong>针对表格内容</strong><br>您可以询问任何问题"),
		textbox=gr.Textbox(placeholder="请在此输入您的问题", container=False, scale=7, label='问题', min_width=240, submit_btn='发送'),
		theme="soft",
		# examples=[{'text': '长江流域的小型水库的库容总量是多少？'}, {'text': '那平均值是多少？'}, {'text': '那水库的名称呢？'}, {'text': '换成中型的呢？'}],
		examples=[['长江流域的小型水库的库容总量是多少？',  '水库'],
				 ['那平均值是多少？',  '水库'],
				 ['那水库的名称呢？',  '水库'],
				 ['换成中型的呢？',  '水库']],
		cache_examples=True,
		# 通过additional_inputs参数添加额外的输入控件,也可以是一个参数值
		additional_inputs=
		[
			# gr.Dropdown(choices=['fund', 'car', 'student', 'reservoir', 'car_sales'], value='reservoir', multiselect = False, allow_custom_value= False, label='数据库')
			table_name,
		],
		additional_inputs_accordion="请选择问答数据库"
	)

###########################################################################################################################
# 通过基础组件自定义聊天界面
import gradio as gr
import random
import time


def respond(message, chat_history):
	"""
	用户输入后的回调函数 respond
	参数
	message: 用户此次输入的消息
	history: 一个 list ，代表到那时为止的 list 。每个内部列表由两个 str 组成，代表一对： [user input, bot response] 。
	历史聊天记录，比如 [["use input 1", "assistant output 1"],["user input 2", "assistant output 2"]]

	返回值：
	第一个：设置输入框的值，输入后需要重新清空，所以返回空字符串，
	第二个：最新的聊天记录
	"""
	bot_message = f"你好：{message}"
	chat_history.append((message, bot_message))
	return "", chat_history

with gr.Blocks() as demo:
	chatbot = gr.Chatbot()  # 对话框
	msg = gr.Textbox(placeholder='请输入你的问题', submit_btn='发送', label='问答')  # 输入文本框
	clear = gr.ClearButton([msg, chatbot])  # 清除按钮

	# 绑定输入框内的回车键的响应函数
	msg.submit(respond, [msg, chatbot], [msg, chatbot])

###########################################################################################################################
# 自定义布局输入组件方面
import gradio as gr  # 导入gradio库，用于创建交互式Web应用
import time  # 导入time模块，用于实现延迟效果


# 定义一个函数，用来逐字输出响应消息
def echo(message, history, system_prompt, tokens):
	# 格式化响应消息，包含系统提示和用户消息
	response = f"系统提示: {system_prompt}\n 消息: {message}."
	# 根据response的长度和tokens确定循环的次数，逐个字符输出
	for i in range(min(len(response), int(tokens))):
		time.sleep(0.05)  # 每输出一个字符后暂停0.05秒，模拟打字效果
		yield response[:i + 1]  # 逐渐输出，每次迭代输出一个字符多于上一次


# 使用gr.Blocks创建一个可视化布局
with gr.Blocks() as demo:
	# 创建一个文本输入框，用于输入系统提示
	system_prompt = gr.Textbox("你是一个全能的AI.", label="系统提示")
	# 创建一个滑动条，用于选择tokens的数量，但不直接在界面中渲染显示
	slider = gr.Slider(10, 100, render=False)

	# 创建一个聊天界面，将echo函数作为处理函数，并通过additional_inputs添加上面创建的输入控件
	gr.ChatInterface(
		echo, additional_inputs=[system_prompt, slider]
	)


###########################################################################################################################

def main():
	# 启动Gradio界面
	demo.launch(server_name='0.0.0.0',
				server_port=8082, share=True)



if __name__ == "__main__":
	main()
