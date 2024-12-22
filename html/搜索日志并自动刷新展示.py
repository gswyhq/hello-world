from flask import Flask, request, render_template_string
import re

'''
URL 参数解析：我们使用 request.args.get('trace_id') 来获取 URL 中的 trace_id 参数。
正则表达式匹配：使用 re.search 来查找每一行中是否包含指定格式的 trace_id。
逻辑控制：
当找到与请求的 trace_id 匹配的行时，开始记录这些行。
如果在之后的行中发现了不同的 trace_id，则停止记录。
如果一行不包含 trace_id 但是前一行的 trace_id 和请求的一致，那么也记录这一行。
自动刷新：页面会每5秒自动刷新一次，并保持查询相同的 trace_id。

http://localhost:7860/logs/?trace_id=supersonic_a28405686ed14488b2d457db76cf4504
'''
app = Flask(__name__)

# 配置要读取的日志文件路径
LOG_FILE_PATH = 'serviceinfo.chat.log'


@app.route('/logs/')
def logs():
    trace_id = request.args.get('trace_id')

    if not trace_id:
        return "No trace_id provided.", 400

    log_content = ''
    capturing = False
    try:
        with open(LOG_FILE_PATH, 'r') as file:
            for line in file:
                # 查找当前行是否包含指定的 trace_id
                match = re.search(r'\[(?P<trace_id>supersonic_[a-f0-9]{32})\]', line)
                if match:
                    current_trace_id = match.group('trace_id')
                    if current_trace_id == trace_id:
                        capturing = True
                        log_content += line
                    elif capturing:
                        # 如果遇到了不同的 trace_id，则停止读取
                        break
                elif capturing:
                    # 如果当前没有其他 trace_id 并且之前已经找到匹配的 trace_id，则继续添加到输出
                    log_content += line

    except Exception as e:
        log_content = f"Error reading log file: {str(e)}"

    # 返回带有内联 JavaScript 的 HTML 内容以实现实时更新
    return render_template_string('''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Log Viewer by Trace ID</title>
    <script type="text/javascript">
        function autoRefresh() {
            location.href = '?trace_id={{ trace_id }}';
        }
        window.onload = function() {
            setInterval('autoRefresh()', 5000); // 每5秒刷新一次页面
        }
    </script>
</head>
<body>
    <h1>Log Viewer for Trace ID: {{ trace_id }}</h1>
    <pre>{{ log_content }}</pre>
</body>
</html>
''', log_content=log_content, trace_id=trace_id)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860)