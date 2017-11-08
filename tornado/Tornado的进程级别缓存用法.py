#!/usr/bin/python3
# coding: utf-8

import tornado.httpserver
import tornado.ioloop
import tornado.options
import tornado.httpclient
import tornado.web
import sys, os
from tornado.options import define, options

#####
# 由于PageCount是全局的，跟随进程实例化，所以可以很灵活在各个实例之间传递数据。
class PageCounter(object):
    count = 0
####

#PageCounter = 0 这种采用全局变量的方法被证实无法实现。

class MainPage(tornado.web.RequestHandler):

    def get(self):
        PageCounter.count += 1
        self.finish('Hello World!')

class GetPageView(tornado.web.RequestHandler):
    def get(self):
        sPageView = str(PageCounter.count)
        self.write(sPageView)

def main(port):
    os_fork = os.fork()
    if os_fork != 0:
        # 从fork()函数原型来看，它也属于一个内建函数。 ;子进程永远返回0，而父进程返回子进程的ID。
        exit()
    define("port", default=port, help="run on the given port", type=int)
    application = tornado.web.Application([
        ("/", MainPage),
        ("/get", GetPageView),
    ])
    http_server = tornado.httpserver.HTTPServer(application)
    http_server.listen(options.port)
    tornado.ioloop.IOLoop.instance().start()

if __name__ == "__main__":

    if len (sys.argv) > 1:
        port = sys.argv[1]
    else:
        port = 8001
    main(port)