#!/usr/bin/python3
# coding: utf-8

import tornado.web
import tornado.ioloop
import time
import sys
import functools
import tornado.concurrent

from tornado.httpserver import HTTPServer

from concurrent.futures import ThreadPoolExecutor

class FutureHandler(tornado.web.RequestHandler):
    executor = ThreadPoolExecutor(10)

    @tornado.web.asynchronous
    @tornado.gen.coroutine
    def get(self, *args, **kwargs):

        url = time.time()
        tornado.ioloop.IOLoop.instance().add_callback(functools.partial(self.ping, url))

        print("do something others")

        self.write("123456")
        self.finish('It works')

        sys.stdout.flush()


    @tornado.concurrent.run_on_executor
    def ping(self, url):
        for i in range(10):
            time.sleep(1)
            print("n: {}, {}".format(url, i))
        return 'after'


# @profile
def make_app():
    return tornado.web.Application([
        (r'/async', FutureHandler), # 异步训练
        ],
        # debug=True
    )


if __name__ == "__main__":
    application = make_app()
    myserver = HTTPServer(application)
    application.listen(8000)
    print('server is running....!!')
    tornado.ioloop.IOLoop.current().start()

# curl -XGET "http://localhost:8000/async" 请求能快速有返回值，但后台还在执行ping