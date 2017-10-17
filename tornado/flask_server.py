#!/usr/bin/python3
# coding: utf-8

import json
import logging
from flask import Flask, request
from main2 import test_produce_answer

app = Flask(__name__)

from functools import wraps
from flask import make_response


def allow_cross_domain(fun):
    """
    允许跨越访问
    :param fun:
    :return:
    """
    @wraps(fun)
    def wrapper_fun(*args, **kwargs):
        rst = make_response(fun(*args, **kwargs))
        rst.headers['Access-Control-Allow-Origin'] = '*'
        rst.headers['Access-Control-Allow-Methods'] = 'PUT,GET,POST,DELETE'
        allow_headers = "Referer,Accept,Origin,User-Agent"
        rst.headers['Access-Control-Allow-Headers'] = allow_headers
        return rst
    return wrapper_fun


@app.route("/hello", methods=['GET', 'POST'])
def hello():
    if request.method == 'POST':
        uid = request.form['uid'] or ''
        text = request.form['text'] or ''
    else:
        uid, text = '', ''
    return "Hello World!{}{}".format(uid, text)

@app.route("/yhb", methods=['GET', 'POST'])
@allow_cross_domain
def yhb():
    _ip = request.remote_addr or ''
    logging.info("请求的主机ip: {}".format(_ip))
    if request.method == 'POST':
        uid = request.form.get('uid', default=_ip) or _ip # 没有uid时，默认为机器的ip
        text = request.form.get('text', '')
    elif request.method == 'GET':
        uid = request.args.get('uid', _ip) or _ip
        text = request.args.get('text', '')
    else:
        result = '请求方式应该为post, 不应该为：{}'.format(request.method)
        data = {'code': 200, 'text': result}
        return json.dumps(data, ensure_ascii=False)
    if uid and text:
        logging.info("请求的参数，uid: {} ,问题： {}".format(uid, text))
        answer_list = test_produce_answer(text, uid)
        data = {'code': 200, 'text': '\n'.join(answer_list), 'uid': uid}
        return json.dumps(data, ensure_ascii=False)
    else:
        result = '请求的参数不对'
        data = {'code': 200, 'text': result}
        return json.dumps(data, ensure_ascii=False)

def main():
    logging.basicConfig(level=logging.DEBUG,
                format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                datefmt='%a, %d %b %Y %H:%M:%S',
                filename='yhb.log',
                filemode='w')
    app.run(host='0.0.0.0', port='5000')

if __name__ == '__main__':
    main()

# print("request.form: {}".format(request.form))
# data_form = {}
# for fieldname, value in request.form.items():
#     data_form[fieldname] = value
#
# print("data_form: {}".format(data_form))
# print("field_question_list: {}".format(request.form.getlist("field_question_list")))
# print("request.args: {}".format(request.args))
# print("request.values: {}".format(request.values))
# print("request.data: {}".format(request.data))
# print("request.json: {}".format(request.json))
# print("request.get_json(): {}".format(request.get_json()))
# print("jsonify(request.form): {}".format(jsonify(request.form)))
#
# print("request.get_json(force=True): {}".format(request.get_json(force=True)))