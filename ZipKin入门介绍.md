
# ZipKin入门介绍

Zipkin是一款开源的分布式实时数据追踪系统（Distributed Tracking System），基于 Google Dapper的论文设计而来，由 Twitter 公司开发贡献。
其主要功能是聚集来自各个异构系统的实时监控数据。

Zipkin Server主要包括四个模块：
（1）Collector 接收或收集各应用传输的数据
（2）Storage 存储接受或收集过来的数据，当前支持Memory，MySQL，Cassandra，ElasticSearch等，默认存储在内存中。
（3）API（Query） 负责查询Storage中存储的数据，提供简单的JSON API获取数据，主要提供给web UI使用
（4）Web 提供简单的web界面

ZipKin几个概念
    在追踪日志中，有几个基本概念spanId、traceId、parentId
    traceId：用来确定一个追踪链的16字符长度的字符串，在某个追踪链中保持不变。
    spanId：区域Id，在一个追踪链中spanId可能存在多个，每个spanId用于表明在某个服务中的身份，也是16字符长度的字符串。
    parentId：在跨服务调用者的spanId会传递给被调用者，被调用者会将调用者的spanId作为自己的parentId，然后自己再生成spanId。

# 运行ZipKin服务
* 快速开始(数据存储在内存中)

`my@192.168.3.133:~$ docker run --rm -p 9411:9411 openzipkin/zipkin:2.14.0`

浏览器打开`http://192.168.3.133:9411/` 可以查看服务及依赖


* 数据存储在MySQL中
```shell
my@192.168.3.133:~$ git clone https://github.com/openzipkin/docker-zipkin.git
my@192.168.3.133:~$ cd docker-zipkin
my@192.168.3.133:~/docker-zipkin$ docker-compose -f docker-compose.yml up -d

my@192.168.3.133:~/docker-zipkin$ docker-compose -f docker-compose.yml ps
          Name                        Command               State                Ports              
----------------------------------------------------------------------------------------------------
dependencies               crond -f                         Up                                      
grafana                    /run.sh                          Up      0.0.0.0:3000->3000/tcp          
mysql                      /bin/sh -c /mysql/run.sh         Up      0.0.0.0:3306->3306/tcp          
prometheus                 /bin/prometheus --config.f ...   Up      0.0.0.0:9090->9090/tcp          
setup_grafana_datasource   /entrypoint.sh /create.sh        Up                                      
zipkin                     /busybox/sh run.sh               Up      9410/tcp, 0.0.0.0:9411->9411/tcp

* 其中 MySQL的用户名/密码为：zipkin/zipkin
* https://github.com/openzipkin/docker-zipkin/blob/master/mysql/configure.sh#L31
```

* 数据存储在elasticsearch中
```shell
my@192.168.3.133: ~$ git clone https://github.com/openzipkin/docker-zipkin.git
my@192.168.3.133: ~$ cd docker-zipkin
my@192.168.3.133: ~/docker-zipkin$ docker-compose -f docker-compose.yml -f docker-compose-elasticsearch.yml up -d 

* 用elasticsearch做数据存储时，依赖关系表未能正常显示，原因不明；
```

# 使用
主要是客户端，及服务端，都需要对ZipKin服务进行相应的交互，下面是个示例：
启动3个服务，调用关系如下：client.py 分别调用 flask_server.py、 tornado_server.py

client.py  
```python
#!/usr/bin/python3
# coding: utf-8

import requests
from flask import Flask
from py_zipkin.zipkin import zipkin_span,create_http_headers_for_new_span
import time

app = Flask(__name__)

def http_transport(encoded_span):
    body=encoded_span
    zipkin_url="http://192.168.3.133:9411/api/v1/spans"
    headers = {"Content-Type": "application/x-thrift"}

    r=requests.post(zipkin_url, data=body, headers=headers)
    print(type(encoded_span))
    print(encoded_span)
    print(body)
    print(r)
    print(r.content)

@zipkin_span(
        service_name='webapp',
        span_name='index',
        transport_handler=http_transport,
        port=5000,
        host='192.168.3.164',
        sample_rate=100, #0.05, # Value between 0.0 and 100.0
    )
def do_stuff():
    # time.sleep(2)
    headers = create_http_headers_for_new_span()
    # print('headers=', headers)
    # headers = {}
    ret = requests.post('http://192.168.3.164:6000/service1/', json={'question': '你好'}, headers=headers)

    print(ret.json())
    return 'OK'




@app.route('/')
def index():
    # with zipkin_span(
    #     service_name='webapp',
    #     span_name='index',
    #     transport_handler=http_transport,
    #     port=5000,
    #     host='192.168.3.164',
    #     sample_rate=100, #0.05, # Value between 0.0 and 100.0
    # ):
        # with zipkin_span(service_name='webapp', span_name='do_stuff'):
    do_stuff()
        # time.sleep(1)
    return 'OK', 200

if __name__=='__main__':
    app.run(host="0.0.0.0",port=5000,debug=True)
```

flask_server.py  
```python
#!/usr/bin/python3
# coding: utf-8

from flask import request, jsonify
import requests
from flask import Flask
from py_zipkin.zipkin import zipkin_span, ZipkinAttrs

app = Flask(__name__)

def http_transport(encoded_span):

    body=encoded_span
    zipkin_url="http://192.168.3.133:9411/api/v1/spans"
    headers = {"Content-Type": "application/x-thrift"}
    requests.post(zipkin_url, data=body, headers=headers)


def do_stuff():
    # with zipkin_span(service_name='service1', span_name='service1_db_search'):
    db_search()
    return 'OK'



def db_search():

    return {"data": "查询成功！"}


@app.route('/service1/', methods=['GET', 'POST'])
def index():
    with zipkin_span(
        service_name='flask_server',
        zipkin_attrs=ZipkinAttrs(
            trace_id=request.headers.get('X-B3-TraceID'),
            span_id=request.headers.get('X-B3-SpanID'),
            parent_span_id=request.headers.get('X-B3-ParentSpanID'),
            flags=request.headers.get('X-B3-Flags'),
            is_sampled=request.headers.get('X-B3-Sampled'),
        ),
        span_name='index_service1',
        transport_handler=http_transport,
        port=6000,
        host='192.168.3.164',
        sample_rate=100, #0.05, # Value between 0.0 and 100.0
    ):
        # with zipkin_span(service_name='service1', span_name='service1_do_stuff'):
        do_stuff()
        
    return jsonify({'a': 1, 'b': 2})

if __name__=='__main__':
    app.run(host="0.0.0.0",port=6000,debug=True)

```


tornado_server.py
```python
#!/usr/lib/python3
# coding = utf-8

import json
import sys
import requests
import tornado
import traceback
import tornado.ioloop
import tornado.web, tornado.gen
from tornado import gen
from tornado.httpserver import HTTPServer
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from tornado.concurrent import run_on_executor

from py_zipkin.zipkin import zipkin_span,ZipkinAttrs

PORT = 6000
# executor = concurrent.futures.ThreadPoolExecutor(2)

# 此函数不能放在MainHandler类下面，否则不能使用, 原因不明；
def get_answer(_ip, data):

    result = {
        'code': 0,
        "msg": "请求成功！",
        "data": data
    }

    return result


def http_transport(encoded_span):
    body=encoded_span
    zipkin_url="http://192.168.3.133:9411/api/v1/spans"
    headers = {"Content-Type": "application/x-thrift"}

    requests.post(zipkin_url, data=body, headers=headers)

class MainHandler(tornado.web.RequestHandler):
    executor = ThreadPoolExecutor()

    @gen.coroutine
    def get(self):
        try:
            print('request.body : %s' % self.request.body)
            _ip = self.request.remote_ip
            data = self.request.arguments
            result = yield self._get_answer(_ip=_ip, data=data)
            result_str = json.dumps(result, ensure_ascii=False)
            self.write(result_str)

        except Exception as e:
            print("解析参数出错： {}".format(e))
            print("错误详情： {}".format(traceback.print_exc()))
            result = {'code': 1, "msg": "请求出错！"}
            result_str = json.dumps(result, ensure_ascii=False)
            self.write(result_str)

        self.write("\n")
        sys.stdout.flush()

    @gen.coroutine
    def post(self):
        try:
            print('request.body : %s' % self.request.body)
            _ip = self.request.remote_ip
            _ip = self.request.headers.get('X-Forwarded-For') or self.request.headers.get(
                'X-Real-IP') or _ip  # 有可能是nginx代理； proxy_set_header X-Forwarded-For  $remote_addr;
            body_data = self.request.body
            if isinstance(body_data, bytes):
                body_data = self.request.body.decode('utf8', errors='ignore')
            data = json.loads(body_data)
            result = yield self._get_answer(_ip=_ip, data=data)
            result_str = json.dumps(result, ensure_ascii=False)
            self.write(result_str)

        except Exception as e:
            print("解析参数出错： {}".format(e))
            print("错误详情： {}".format(traceback.print_exc()))
            result = {'code': 1, "msg": "请求出错！"}
            result_str = json.dumps(result, ensure_ascii=False)
            self.write(result_str)

        self.write("\n")
        sys.stdout.flush()

    @run_on_executor
    def _get_answer(self, _ip, data):
        with zipkin_span(
                service_name='tornado_server',
                zipkin_attrs=ZipkinAttrs(
                    trace_id=self.request.headers.get('X-B3-TraceID'),
                    span_id=self.request.headers.get('X-B3-SpanID'),
                    parent_span_id=self.request.headers.get('X-B3-ParentSpanID'),
                    flags=self.request.headers.get('X-B3-Flags'),
                    is_sampled=self.request.headers.get('X-B3-Sampled'),
                ),
                span_name='index_service2',
                transport_handler=http_transport,
                port=6000,
                host='192.168.3.164',
                sample_rate=100,  # 0.05, # Value between 0.0 and 100.0
        ):
            return get_answer(_ip, data)

# @profile
def make_app():
    return tornado.web.Application([
        (r"/service1/", MainHandler),
        ],
        cookie_secret="61oETzKXQAGaYdkL5gEmGeJJFuYh7EQnp2XdTP1o/Vo="
    )

def main():
    application = make_app()
    myserver = HTTPServer(application)
    application.listen(PORT)
    print('server is running....!!')
    tornado.ioloop.IOLoop.current().start()


if __name__ == "__main__":
    main()
```

在本地机器（这里是192.168.3.164）上，运行 client.py + flask_server.py 或者 client.py + tornado_server.py
命令行运行`curl http://192.168.3.164:5000/ `, 浏览器打开`http://192.168.3.133:9411`查看调用链分析结果。


