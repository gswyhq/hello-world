#!/usr/bin/python3
# coding: utf-8

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from tornado.concurrent import run_on_executor
import time
import tornado
from tornado.web import RequestHandler
from tornado import gen, web
import tornado.ioloop
import traceback

# 此函数不能放在NoBlockingHnadler类下面，否则不能使用, 原因不明；
def doing(s):
    print('xiumian--{}'.format(s))
    time.sleep(s)
    return "等待时间： {}".format(s)

class NoBlockingHnadler(RequestHandler):
    executor = ProcessPoolExecutor(max_workers=4)   # 新建`max_workers`个进程池，总进程数是“max_workers+1”，静态变量，属于类，所以全程只有这“max_workers+1”个进程，不需要关闭，如果放在__init__中，则属于对象，每次请求都会新建pool，当请求增多的时候，会导致进程变得非常多，这个方法不可取
    print('开始{}'.format(executor._processes))

    @gen.coroutine
    def get(self, *args, **kwargs):
        print('开始{}'.format(self.executor._processes))
        a = yield self.executor.submit(doing, 8)
        print('进程 %s'  % self.executor._processes)
        self.write(str(a))
        print('执行完毕{}'.format(a))

class NoBlockingHnadler2(RequestHandler):
    executor = ThreadPoolExecutor(4)
    @web.asynchronous
    @gen.coroutine
    def get(self, *args, **kwargs):
        try:
            a = yield self.doing2(8)
            self.write(str(a))
            print('执行完毕{}'.format(a))
        except Exception as e:
            print(e)
            print(traceback.format_exc())

    @run_on_executor  #tornado 另外一种写法，需要在静态变量中有executor的线程池变量
    def doing2(self, s):
        print('实现方式2--{}'.format(s))
        time.sleep(s)
        return "等待时间： {}".format(s)

class MainHandler(tornado.web.RequestHandler):
    @tornado.web.asynchronous
    def get(self):
        self.write("Hello, world: {}".format(time.time()))
        self.finish()

    @tornado.web.asynchronous
    def post(self):
        self.write("Hello, world: {}".format(time.time()))
        self.finish()

def make_app():
    return tornado.web.Application([
        (r"/", MainHandler),
        (r"/noblock", NoBlockingHnadler),
        (r"/noblock2", NoBlockingHnadler2),
    ], autoreload=True)

def main():
    app = make_app()
    app.listen(1180)
    tornado.ioloop.IOLoop.current().start()

if __name__ == '__main__':
    main()