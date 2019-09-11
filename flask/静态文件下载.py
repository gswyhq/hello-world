
import traceback
from flask import Flask, request
import json

from flask import send_file, send_from_directory
import os
from flask import make_response

app = Flask(__name__)

# 跨域支持
def after_request(resp):
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp

app.after_request(after_request)

@app.route("/download/<filename>", methods=['GET'])
def download_file(filename):
    # 需要知道2个参数, 第1个参数是本地目录的path, 第2个参数是文件名(带扩展名)
    directory = "./download"  # 假设在当前目录
    response = make_response(send_from_directory(directory, filename, as_attachment=True))
    response.headers["Content-Disposition"] = "attachment; filename={}".format(filename.encode().decode('latin-1'))
    return response

@app.route('/nlp/kg_answer_generation', methods=['POST'])
def upload():
    data = request.get_json(force=True)
    if request.method == 'POST':
        try:
            uid = data.get('uid', '')
            result = {}
            ret_str = json.dumps({"code":200,"msg":"请求成功","data":result},ensure_ascii=False)
            return ret_str
        except Exception as e:
            print("请求出错：{}".format(e))
            print('错误详情：{}'.format(traceback.format_exc()))
            return json.dumps({"code":400,"msg":"请求失败","data":{}},ensure_ascii=False)
    else:
        ret = {
            "code": 1,
            "msg": "不支持的操作"
        }
        return json.dumps(ret, ensure_ascii=False)


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=9627, use_reloader=False)


# curl -XGET http://localhost:9627/download/20190910.log

