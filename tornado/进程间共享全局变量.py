#!/usr/bin/python3
# coding: utf-8

import sys
import os
import traceback
import copy
import json
import tornado
import tornado.escape
import tornado.web
from tornado import gen
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from tornado.concurrent import run_on_executor
from tornado.httpserver import HTTPServer
from multiprocessing import Manager

# 进程数
PROCESS_POOL_COUNT = None

# 线程数
THREAD_POOL_COUNT = None

class GlobalVariable():
    # POOL = 'processpool'
    POOL = 'threadpool'
    count = 0

global_variable = GlobalVariable()

PORT = 8000

manager = Manager()
vip_list = manager.list()
vip_list.append(0)

vip_dict = manager.dict()

vip_dict['global_variable'] = global_variable

# 此函数不能放在MainHandler类下面，否则不能使用, 原因不明；
def get_answer(_ip, data):
    pid = os.getpid() # 是获取的是当前进程的进程号，
    ppid = os.getppid() # 是获取当前进程的父进程的进程

    # 线程间的变量共享
    global_variable.count += 1
    count = global_variable.count

    # vip_dict['global_variable'].count += 1
    # count = vip_dict['global_variable'].count

    # 进程间的变量共享
    # vip_list[0] += 1
    # count = vip_list[0]

    # 不同进程间的变量id虽说一致，但其对应的数据不一致
    vid = id(global_variable)
    vid2 = id(global_variable.count)
    result = {'code': 200, "data": "你好", "pid": pid, "ppid": ppid, "count": count, "vid": vid, "vid2": vid2}
    return result

def check_json_format(raw_msg):
    """
    用于判断一个字符串是否符合Json格式
    :param self:
    :return:
    """
    if isinstance(raw_msg, str):       # 首先判断变量是否为字符串
        try:
            return json.loads(raw_msg, encoding='utf-8')
        except ValueError:
            return False
        # return True
    else:
        return False

class MainHandler(tornado.web.RequestHandler):
    if global_variable.POOL == 'threadpool':
        print("线程数： {}".format(THREAD_POOL_COUNT))
        executor = ThreadPoolExecutor(THREAD_POOL_COUNT)

    elif global_variable.POOL == 'processpool':
        print("进程数： {}".format(PROCESS_POOL_COUNT))
        executor = ProcessPoolExecutor(PROCESS_POOL_COUNT)  # 新建`max_workers`个进程池，总进程数是“max_workers+1”，静态变量，属于类，所以全程只有这“max_workers+1”个进程，不需要关闭，如果放在__init__中，则属于对象，每次请求都会新建pool，当请求增多的时候，会导致进程变得非常多，这个方法不可取

    def get(self):
        self.write("None")
        print("request.uri : {}".format(self.request.uri))

    @gen.coroutine
    def post(self):
        try:
            print('request.body : %s' % self.request.body)
            _ip = self.request.remote_ip
            data = self.args_parse
            if not data:
                self.write("\n")
            else:
                if global_variable.POOL == 'threadpool':
                    result = yield self._get_answer(_ip=_ip, data=data)
                elif global_variable.POOL == 'processpool':
                    result = yield self.executor.submit(get_answer, _ip=_ip, data=data)
                else:
                    result = get_answer(_ip=_ip, data=data)
                print("返回数据：{}".format(result))
                self.write(result)
                self.write("\n")
        except Exception as e:
            print("解析参数出错： {}".format(e))
            print("错误详情： {}".format(traceback.print_exc()))
            pid = os.getpid()  # 是获取的是当前进程的进程号，
            ppid = os.getppid()  # 是获取当前进程的父进程的进程
            result = {'code': 200, "data": "你好", "pid": pid, "ppid": ppid}
            self.write(result)
            self.write("\n")
        sys.stdout.flush()

    @property
    def args_parse(self):
        """解析传入的参数"""
        from tornado.httputil import parse_body_arguments

        args_data = {}
        for name, values in self.request.arguments.items():
            json_name = check_json_format(name)
            if json_name:
                args_data.update(json_name)
            else:
                args_data[name] = values[0].decode('utf8').strip()

        body_data = self.request.body
        if not args_data and body_data:
            if isinstance(body_data, bytes):
                body_data = self.request.body.decode('utf8')
            args_data = tornado.escape.json_decode(body_data)

        print("传入参数: {}".format(args_data))
        return args_data

    def set_default_headers(self):
        self.set_header("Access-Control-Allow-Origin", "*")  # 允许跨域访问
        self.set_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.set_header("Content-Type", "text/plain; charset=UTF-8")
        self.set_header("Access-Control-Allow-Headers", "Origin, X-Requested-With, Content-Type, Accept")

    def options(self):
        # no body
        self.set_status(204)
        self.finish()

    @run_on_executor
    def _get_answer(self, _ip, data):
        return get_answer(_ip, data)


def make_app():
    return tornado.web.Application([
        (r"/", MainHandler),
        ],
    )


if __name__ == "__main__":
    application = make_app()
    myserver = HTTPServer(application)
    application.listen(PORT)
    print('server is running....!!')
    tornado.ioloop.IOLoop.current().start()

