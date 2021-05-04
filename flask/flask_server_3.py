#!/usr/bin/python3
# coding: utf-8
import json
from flask import Flask, request, jsonify
from flask import Response

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False  # 保证用jsonify方法返回给客户端的不是乱码

@app.route('/', methods=['POST','GET' ])
def hello_world():

    # args = request.args
    # print(args)
    # print(dict(request.form))
    # print(request.values)
    args = request.get_json()
    if not args:
        args = request.args
        print(request.args.get('text', ''))
    data = {"code":200, "msg":"请求成功！"}
    data.update(args)
    r = Response(response=json.dumps(data, ensure_ascii=False), status=200, mimetype="application/json; charset=UTF-8")
    return r
    # return json.dumps(data, ensure_ascii=False) + '\n'
    # return jsonify(data)
    
def main():
    app.run(host='0.0.0.0', port='8888')

if __name__ == '__main__':
    main()

# http://docs.jinkan.org/docs/flask/quickstart.html#a-minimal-application
# gswewf@gswewf-PC:~$ curl -XPOST 192.168.3.103:8888/ -d '{"message": "12345"}'
# gswewf@gswewf-PC:~$ curl 192.168.3.103:8888/?text=123
# from urllib.parse import quote
# quote('你好')
# Out[16]: '%E4%BD%A0%E5%A5%BD'
# curl -XGET 192.168.3.103:8888/?text=%E4%BD%A0%E5%A5%BD
# gswewf@gswewf-PC:~$ curl -XPOST 192.168.3.103:8888/ -d '{"message": "你好"}' -H "Content-Type: application/json"

