#!/usr/bin/python3
# coding: utf-8
import json
from flask import Flask, request, jsonify
from flask import Response

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False  # 保证用jsonify方法返回给客户端的不是乱码

# 有时我们需要在不重启服务器的情况下加载并执行新文件。
# 先启动服务，在不重启服务器的情况下加载并执行新文件
# 前提条件是服务启动执行必须先有对应的支持代码部分

# 第一步，准备好新文件my_config.py内容：
'''
def mymod(*args, **kwargs):
   print("我是新文件的数据")
'''
# 第二步，在get1接口函数中添加新文件加载处理函数 load_sour()
# 第三步，启动服务后，给Python服务发送命令，使之加载对应的py文件并执行对应的函数；

def load_sour(funtion_name, funtion_path, *args, **kwargs):
    '''
    加载新文件并执行对应函数；
    :param funtion_name: 需要执行的函数名称
    :param funtion_path: 需要加载的py文件路径， 如：./abc/my_config.py
    :param args: 需要执行的函数的列表参数
    :param kwargs: 需要执行的函数的字典参数
    :return: 
    '''
    print("执行：load_sour")
    import importlib
    a = importlib.machinery.SourceFileLoader(funtion_name, funtion_path).load_module()
    getattr(a, funtion_name)(*args, **kwargs)

@app.route('/get1', methods=['GET', 'POST'])
def get_handler():
    """
    请求示例：
    curl localhost:8888/get1 -d '{"funtion_name": "mymod", "funtion_path": "./abc/my_config.py", "args": [], "kwargs": {} }'  -H 'Content-Type: application/json'
    
    :return: 
    """
    print("传入参数：", request.json)  # curl请求时，需指明; -H 'Content-Type: application/json'
    data = request.json
    funtion_name = data.get("funtion_name", "")
    funtion_path = data.get("funtion_path", "")
    args = data.get("args", [])
    kwargs = data.get("kwargs", {})

    load_sour(funtion_name, funtion_path, *args, **kwargs)

    ret = {
        "code":0,
        "msg":"success"
        }
    return Response(response=json.dumps(ret, ensure_ascii=False), status=200, mimetype="application/json; charset=UTF-8")

def main():
    app.run(host='0.0.0.0', port='8888')

if __name__ == '__main__':
    main()


# curl localhost:8888/get1 -d '{"funtion_name": "mymod", "funtion_path": "./abc/my_config.py", "args": [], "kwargs": {} }'  -H 'Content-Type: application/json'




