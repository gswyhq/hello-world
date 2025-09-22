#!/usr/bin/env python
# coding=utf-8

import gradio as gr

with gr.Blocks() as demo:
    gr.Markdown("## True Hover Tooltip Demo (using HTML)")

    # 使用 gr.HTML 和 HTML 的 title 属性
    gr.HTML(
        """
        <p>
            将鼠标悬停在 <span title='这是一个由 HTML title 属性生成的真正 Tooltip！' style='color: blue; text-decoration: underline; cursor: help;'>这个文本</span> 上查看提示。
        </p>
        """
    )

    # 你也可以用在 Markdown 中
    gr.Markdown(
        """
        你也可以在 Markdown 中使用 HTML:
        <span title='这是 Markdown 里的 Tooltip' style='cursor: help;'>悬停这里</span>.
        """
    )

    with gr.Row():
        gr.Button("执行主要操作")
        gr.HTML(
            """<span title='点击该按钮将开始处理数据，可能需要几分钟时间。' style='cursor: help; margin-left: 10px;'>❓</span>"""
        )

    with gr.Row():
        gr.HTML(
            """
            <p>
                <span title='用于验证机器人是否正常，若不包含该字符串，则生成报告的时候固定报“机器人出错”' style='color: blue; text-decoration: underline; cursor: help;'>ⓘ</span>
            </p>
            """,
        )
        validation_string2 = gr.Textbox(
            label="验证字符串",
            value="已收到您的问题", scale=95
        )

    with gr.Column():
        gr.HTML(
            """
            <p>
                <span title='检验是否启用了机器人修复功能，若配置了该项，则生成报告的时候，不包含关键词是从该内容之后的答案中进行匹配，而不是在全部答案中进行匹配' style='color: blue; text-decoration: underline; cursor: help;'>❓</span>
            </p>
            """
        )
        repair_keyword2 = gr.Textbox(
            label="机器人修复功能关键词",
            value="尝试进行ai纠正", scale=95
        )


# 启动Gradio应用
demo.launch(server_name='0.0.0.0', server_port=7860, share=True)

def main():
    pass


if __name__ == "__main__":
    main()


###########################################################################################################################################################################################
import gradio as gr
import time

def process_question(question):
    """ 这是一个模拟的后端处理函数。 """
    if not question: # 检查是否是默认值，如果是，也视为未输入有效内容
        if question == "在这里输入您的问题...":
            return "警告：请修改默认内容并输入您的问题！"
        return "警告：您没有输入任何内容！"

    print(f"后台收到的问题是: '{question}'")
    # 模拟一些处理耗时
    time.sleep(1)
    return f"✅ 后台已处理您的问题：**{question}**"

html_content = """
<div id="custom-input-container">
    <label for="custom-input-box" title="这是一个工具提示：请输入您想问的任何问题。">
        <strong>用户名ℹ️</strong>
    </label>
    <!-- 关键点：oninput 调用全局函数 updateGradioInput() -->
    <input type="text" id="custom-input-box" value="在这里输入您的问题..." oninput="updateGradioInput()">
</div>
<style>
    /* 添加一些简单的样式 */
    #custom-input-container {
        padding: 10px;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        background-color: #f9f9f9;
    }
    #custom-input-box {
        width: 98%;
        padding: 8px;
        margin-top: 5px;
        border: 1px solid #ccc;
        border-radius: 4px;
    }
</style>
"""

js_code_v3 = """
window.updateGradioInput = function() {
    // 获取自定义HTML输入框的元素和它的值
    const customInput = document.getElementById('custom-input-box');
    const customValue = customInput.value;

    // 在 Gradio 3.x 中，我们可以直接从 document 中选择元素
    const gradioInput = document.querySelector('#hidden-input-for-js textarea') || document.querySelector('#hidden-input-for-js input');

    // 确保元素被找到
    if (gradioInput) {
        // 将自定义输入框的值同步到隐藏的Gradio Textbox
        gradioInput.value = customValue;

        // 触发一个'input'事件，通知Gradio更新其内部状态
        gradioInput.dispatchEvent(new Event('input', { bubbles: true }));
    } else {
        console.error("Gradio input/textarea element not found within bridge.");
    }
}
"""

with gr.Blocks(js=js_code_v3) as demo:
    gr.Markdown("# 📝 自定义 HTML 输入示例")
    gr.Markdown("下面的输入框完全由自定义HTML和JS实现，它将数据传递给Gradio后端。")

    # 展示我们的自定义HTML界面
    gr.HTML(value=html_content)

    # 创建一个隐藏的Textbox作为数据代理
    hidden_proxy_input = gr.Textbox(
        value="在这里输入您的问题...",
        visible=False,
        elem_id="hidden-input-for-js"
    )

    # 创建提交按钮和输出区域
    with gr.Row():
        submit_btn = gr.Button("提交问题", variant="primary")

    output_display = gr.Markdown(value="--- \n 等待提交...")

    # 设置点击事件
    submit_btn.click(
        fn=process_question,
        inputs=hidden_proxy_input,
        outputs=output_display
    )

if __name__ == "__main__":
    demo.launch(server_name='0.0.0.0', server_port=7860, share=True)


