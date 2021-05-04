#!/usr/bin/python3
# coding: utf-8

from flask import Flask, request
from flask import jsonify


def create_api(functions, port):
    app = Flask(__name__)
    if not isinstance(functions, list):
        functions = [functions]
    print(functions)
    for f in functions:
        def web_api():
            data = request.get_json(force=True)
            return jsonify(f(data))

        app.add_url_rule('/' + f.__name__, f.__name__, web_api, methods=['POST'])
        print('a api http://0.0.0.0:%s/%s create' % (port, f.__name__))
    app.run(debug=False, host='0.0.0.0', port=port)


def your_api(data):
    print('使用your_api解析')
    print(data)
    return data


def your_api_2(data):
    print('使用your_api2解析')
    print(data)
    return data


def main():
    create_api([your_api, your_api_2], 8811)


if __name__ == '__main__':
    main()

# 使用http调用服务
# import requests
# import json
# def api_call(url, data):
#     r = requests.post(url, data = json.dumps(data) )
#     result = r.json()
#     return result
# result = api_call('http://192.168.3.51:8811/your_api', '123456')
# print(result)
# result = api_call('http://192.168.3.51:8811/your_api_2', '123456')
# print(result)
