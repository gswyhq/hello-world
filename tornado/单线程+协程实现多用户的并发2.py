#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import traceback
import sys
import time
import tornado
import tornado.ioloop
import tornado.web, tornado.gen
from tornado import gen
from tornado.httpserver import HTTPServer
from tornado.httpclient import AsyncHTTPClient, HTTPRequest
import logging as logger

NEO4J_HTTP_PORT = 7474
NEO4J_BOLT_PORT = 7687
NEO4J_HOST = 'localhost'
NEO4J_USER = 'neo4j'
NEO4J_PASSWORD = 'gswyhq'
NEO4J_TIMEOUT = (1, 6)  # 超时设置

STANDARD_ORIGIN_DICT = {}

@gen.coroutine
def get_answer(data):
    entity = data.get('entity', {})
    intent = data.get('intent', '')
    entity_value, entity_type = list(entity.items())[0]
    cypher = 'MATCH (n1:{})-[r:`{}`]-(n2) where n1.name="{}"  RETURN DISTINCT  n2.name  as answer limit 3'.format(entity_type, intent, entity_value)
    answer = yield neo4j_search(cypher)
    return answer

class MainHandler(tornado.web.RequestHandler):
    @gen.coroutine
    def post(self):
        try:
            body_data = self.request.body
            if isinstance(body_data, bytes):
                body_data = self.request.body.decode('utf8', errors='ignore')

            data = json.loads(body_data)
            print("传入参数： {}".format(data))
            intent = data.get('intent', '')
            entity = data.get('entity', {})
            assert intent or entity, '【错误】输入参数有误，意图或实体不能同时为空！'

            answer = yield get_answer(data=data)

            result = {"code": 0, "msg": "请求成功！", "answer": answer }

            result_str = json.dumps(result, ensure_ascii=False)
            print("返回数据: {}".format(result_str))
            self.write(result_str)
            self.write("\n")
        except (AssertionError, ValueError) as e:
            ret = {}
            ret['code'] = 1
            ret['msg'] = str(e) if isinstance(e, (AssertionError, ValueError)) else '请求出错'
            ret_str = json.dumps(ret, ensure_ascii=False)
            print('出现错误：{}'.format(ret_str))
            self.write(ret_str)
            self.write("\n")
        except Exception as e:
            print("解析参数出错： {}".format(e))
            print("错误详情： {}".format(traceback.print_exc()))
            ret = {"code": 1, "msg": "系统错误！"}
            ret_str = json.dumps(ret, ensure_ascii=False)
            self.write(ret_str)
            self.write("\n")
        sys.stdout.flush()

@gen.coroutine
def post_statements(statements):
    '''

    {
      "statements" : [ {
        "statement" : "CREATE (n) RETURN id(n)"
      }, {
        "statement" : "CREATE (n {props}) RETURN n",
        "parameters" : {
          "props" : {
            "name" : "My Node"
          }
        }
      },
{
    "statement" : "match (n ) where n.name = {props}.name RETURN n",
    "parameters" : {
      "props" : {
        "name" : "恶性肿瘤"
      }
    }
  }
 ]

    :param statements:
    :return:
    '''
    url = 'http://{host}:{http_port}/db/data/transaction/commit'.format(host=NEO4J_HOST,
                                                                        http_port=NEO4J_HTTP_PORT)
    body = {
        "statements": statements
    }
    start = time.time()
    print('批量查询cypher数量：{}'.format(len(statements)))

    # print("body = ", body)
#        print("url: {}, dict: {}".format(url, [NEO4J_USER,  NEO4J_PASSWORD]))
#     print("""curl {} -u neo4j:gswyhq -H "Content-Type: application/json; charset=UTF-8" -d '{}' """.format(url, json.dumps(body, ensure_ascii=0)))

    # r = requests.post(url, json=body, headers={"Content-Type": "application/json; charset=UTF-8","Connection":"close"},
    #                   auth=(NEO4J_USER, NEO4J_PASSWORD), timeout=NEO4J_TIMEOUT)
    request = HTTPRequest(url, method="POST", headers={"Content-Type": "application/json; charset=UTF-8","Connection":"close"}, body=json.dumps(body, ensure_ascii=False), auth_username=NEO4J_USER, auth_password=NEO4J_PASSWORD,
        connect_timeout=1, request_timeout=6)
    r = yield AsyncHTTPClient().fetch(request)
    cost_time = time.time() - start
    if cost_time>0.5:
        print('costtime: {}, cypher:{}'.format(cost_time, statements))
    # ret = r.json()
    ret = json.loads(r.body.decode())
    errors = ret.get("errors", [])
    if errors:
        print("在neo4j查询出错：{}".format(errors))
    return ret

@gen.coroutine
def neo4j_search(cypher):
    statements = []
    statements.append({
        "statement": cypher
    })
    ret = yield post_statements(statements)
    # {'results': [{'columns': ['answer'], 'data': [{'row': ['附加吉祥安康重疾的宽限期为：60日。'], 'meta': [None]}]}], 'errors': []}
    # {'results': [{'columns': ['answer'], 'data': [{'row': ['附加吉祥安康重疾的宽限期为：60日。'], 'meta': [None]}, {'row': ['您支付首期保险费后，除本合同另有约定外，如果您到期未支付保险费，自保险费约定支付日的次日零时起60日为宽限期。宽限期内发生的保险事故，我们仍会承担保险责任，但在给付保险金时会扣减您欠交的保险费。'], 'meta': [None]}, {'row': ['金佑人生A款2018版的宽限期为60天。'], 'meta': [None]}, {'row': ['暂无附加安心住院医疗宽限期资料，请联系人工客服咨询。'], 'meta': [None]}, {'row': ['福佑安康的宽限期为：60日。'], 'meta': [None]}]}], 'errors': []}
    print("cypher语句：`{}`，在neo4j中查询的结果：`{}`。".format(cypher, ret))

    results = ret.get('results', [])
    if results:
        datas = sum([line['data'] for line in results if line.get('data')], [])
        answer_list = sum([line['row'] for line in datas if line.get('row')], [])
        answer = '、'.join([STANDARD_ORIGIN_DICT.get(item, item) for item in answer_list if item])
    else:
        answer = ''
    return answer


PORT = 8000

class MyFile(tornado.web.StaticFileHandler):

    def set_extra_headers(self, path):
        self.set_header("Cache-control", "no-cache")
        self.set_header("Content-Type", "text/plain; charset=utf-8")  # 若是HTML文件，用浏览器访问时，显示所有的文件内容
        # self.set_header("Content-Type", "text/html; charset=utf-8")  # 若是HTML文件，用浏览器访问时，仅仅显示body部分；


# @profile
def make_app():
    return tornado.web.Application([
        (r"/", MainHandler),
        (r"/api", MainHandler),
        (r"/parser", MainHandler),
        (r"/myfile/(.*)", MyFile, {"path": "./output/"}), # 提供静态文件下载； 如浏览器打开‘http://192.168.3.145:8000/myfile/place.pdf’即可访问‘./output/place.pdf’文件
    ],
        static_path= 'communication',
        static_url_prefix= '/home/',
        # debug=True,
        cookie_secret="61oETzKXQAGaYdkL5gEmGeJJFuYh7EQnp2XdTP1o/Vo="
    )

def main():
    application = make_app()
    myserver = HTTPServer(application)
    application.listen(PORT)
    info_str = "server is running at {}....!!".format(PORT)
    # logger.info(info_str)
    print(info_str)
    tornado.ioloop.IOLoop.current().start()


if __name__ == "__main__":
    main()
    
# curl localhost:8000/ -d '{"entity": {"爱无忧A": "Baoxianchanpin"}, "intent": "投保要求"}'
# {"code": 0, "msg": "请求成功！", "answer": "本产品的投保年龄为：30天至50周岁。保障期间为：至60岁、至70岁、至80岁。缴费方式为：趸缴、年缴。"}    