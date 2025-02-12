
import gradio as gr
import time

def slow_qa(question):
    """模拟一个耗时的问答函数"""
    if 'a' in question:
        time.sleep(10)  # 模拟耗时操作
    return f"Answer to: {question}"

# 创建一个Gradio Blocks实例
with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox(placeholder="Enter your question here...")
    clear = gr.Button("Clear")

    def user(user_message, history):
        return "", history + [[user_message, None]]

    def bot(history):
        user_question = history[-1][0]
        response = slow_qa(user_question)
        history[-1][1] = response
        return history

    # 绑定输入框提交事件到'user'函数，并在完成后调用'bot'函数
    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, chatbot, chatbot)

    # 清除按钮点击事件
    clear.click(lambda: None, None, chatbot, queue=False)

# 启动Gradio应用并启用队列以支持并发请求
demo.queue(
    status_update_rate="auto",
    api_open=False,
    max_size=None,
    default_concurrency_limit=20  # 使用正确的参数名
)
demo.launch()

