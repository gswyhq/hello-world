from flask import Flask, request, render_template_string
import subprocess

app = Flask(__name__)

# 假设这是你的日志文件路径
LOG_FILE_PATH = "serviceinfo.chat.log"
'''需要浏览器中输入url，通过URL参数来传递trace_id，如http://localhost:7860/logs?trace_id=123456,
要求通过url中获取trace_id 再在页面中输入需要展示的行数，再根据trace_id搜索日志文件，并使用grep `trace_id` serviceinfo.chat.log |tail -n 展示对应的内容。'''


# Flask 路由和视图函数
@app.route('/logs/', methods=['GET'])
def logs():
    trace_id = request.args.get('trace_id')
    if not trace_id:
        return "Please provide a trace_id in the URL parameters."

    # 渲染一个包含输入框的表单，用于用户输入想要展示的行数
    html_content = """
    <html>
    <head>
        <title>Log Viewer</title>
    </head>
    <body>
        <h1>View Logs for Trace ID: {{ trace_id }}</h1>
        <form method="post" action="/display_logs">
            <label for="num_lines">Number of lines to display:</label>
            <input type="number" id="num_lines" name="num_lines" min="1" required>
            <input type="hidden" name="trace_id" value="{{ trace_id }}">
            <button type="submit">Display Logs</button>
        </form>
    </body>
    </html>
    """
    return render_template_string(html_content, trace_id=trace_id)


@app.route('/display_logs', methods=['POST'])
def display_logs():
    trace_id = request.form['trace_id']
    num_lines = int(request.form['num_lines'])

    try:
        # 使用subprocess模块执行grep和tail命令
        command = f"grep '{trace_id}' {LOG_FILE_PATH} | tail -n {num_lines}"
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
        log_lines = result.stdout.splitlines()

        # 渲染日志内容
        html_content = """
        <html>
        <head>
            <title>Log Output</title>
        </head>
        <body>
            <h1>Logs for Trace ID: {{ trace_id }}</h1>
            <pre>{{ log_lines }}</pre>
        </body>
        </html>
        """
        return render_template_string(html_content, trace_id=trace_id, log_lines='\n'.join(log_lines))
    except subprocess.CalledProcessError as e:
        return f"Error executing command: {e}"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
