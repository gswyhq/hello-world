#!/usr/bin/env python
# coding=utf-8

import gradio as gr

blocks = gr.Blocks()

with blocks as demo:
    subject = gr.Textbox(placeholder="主语")
    verb = gr.Radio(["ate", "loved", "hated", 'eta', 'devol', 'detah'])
    object = gr.Textbox(placeholder="宾语")

    with gr.Row():
        btn = gr.Button("创建句子")
        reverse_btn = gr.Button("反转句子")
        foo_bar_btn = gr.Button("追加 foo")
        reverse_then_to_the_server_btn = gr.Button(
            "反转句子并发送到服务器"
        )

    def sentence_maker(w1, w2, w3):
        return f"{w1} {w2} {w3}"


    output1 = gr.Textbox(label="输出1")
    output2 = gr.Textbox(label="第二个词")
    output3 = gr.Textbox(label="反转词")
    output4 = gr.Textbox(label="前端处理后发送到后端")

    btn.click(sentence_maker, [subject, verb, object], output1)
    reverse_btn.click(
        None, [subject, verb, object], output2, js="(s, v, o) => o + ' ' + v + ' ' + s"
    )
    verb.change(lambda x: x, verb, output3, js="(x) => [...x].reverse().join('')")
    foo_bar_btn.click(None, [], subject, js="(x) => x + ' foo'")

    reverse_then_to_the_server_btn.click(
        sentence_maker,
        [subject, verb, object],
        output4,
        js="(s, v, o) => [s, v, o].map(x => [...x].reverse().join(''))",
    )

def main():
    # 启动Gradio界面
    demo.launch(server_name='0.0.0.0', server_port=8082, share=True)

if __name__ == "__main__":
    main()