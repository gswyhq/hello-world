#!/usr/bin/env python
# coding=utf-8

import gradio as gr

# 第三方代码嵌入：
# 来自：https://github.com/1Panel-dev/MaxKB
# 全屏模式：
# <iframe
# src="http://192.168.3.105:8080/ui/chat/fb7b4eb8dc90c18b"
# style="width: 100%; height: 100%;"
# frameborder="0"
# allow="microphone">
# </iframe>

# 浮窗模式：
# <script
# async
# defer
# src="http://192.168.3.105:8080/api/application/embed?protocol=http&host=192.168.3.105:8080&token=fb7b4eb8dc90c18b">
# </script>

# 浮窗模式：
with gr.Blocks(head="""
    <script
        async
        defer
        src="http://192.168.3.105:8080/api/application/embed?protocol=http&host=192.168.3.105:8080&token=fb7b4eb8dc90c18b">
    </script>
""") as demo:
    gr.HTML('''<iframe
    src="http://192.168.3.105:8080/ui/chat/fb7b4eb8dc90c18b"
    style="width: 100%; height: 800px;"
    frameborder="0"
    allow="microphone">
    </iframe>
    ''')

def main():
    demo.queue(status_update_rate="auto", default_concurrency_limit=120, max_size=120)
    demo.launch(server_name='0.0.0.0', server_port=7860, share=True)



if __name__ == "__main__":
    main()


