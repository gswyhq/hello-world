#!/usr/bin/python3
# coding: utf-8

import tornado.httpserver
import tornado.ioloop
import tornado.options
import tornado.web
import tornado.gen
import tornado.httpclient
import tornado.concurrent
import tornado.ioloop

from tornado.concurrent import run_on_executor
from concurrent.futures import ThreadPoolExecutor

import time

from tornado.options import define, options
define("port", default=28000, help="run on the given port", type=int)

class SleepHandler(tornado.web.RequestHandler):

    executor = ThreadPoolExecutor(2)

    @tornado.web.asynchronous
    @tornado.gen.coroutine
    def get(self):
        # yield tornado.gen.Task(tornado.ioloop.IOLoop.instance().add_timeout, time.time() + 5)
        # self.write("when i sleep 5s")
        result = yield self.sleep_time()
        self.write(result)

    # @tornado.gen.coroutine
    @run_on_executor
    def sleep_time(self):
        time.sleep(10)
        return 'aaadaaaaaaaa'

class JustNowHandler(tornado.web.RequestHandler):
    def get(self):
        self.write("i hope just now see you")

if __name__ == "__main__":
    tornado.options.parse_command_line()
    app = tornado.web.Application(handlers=[
            (r"/sleep", SleepHandler), (r"/justnow", JustNowHandler)])
    http_server = tornado.httpserver.HTTPServer(app)
    http_server.listen(options.port)
    tornado.ioloop.IOLoop.instance().start()
