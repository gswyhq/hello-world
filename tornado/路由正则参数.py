#!/usr/bin/python3
# coding: utf-8

import tornado.web
from tornado import gen
import tornado
import tornado.ioloop
import tornado.web
from tornado.httpserver import HTTPServer

class MainHandler(tornado.web.RequestHandler):

    def initialize(self, database='', **kwargs):
        self.database = database

    def get(self, *args, **kwargs):
        self.write("database: " + self.database)
        print('args: ', args)
        print("kwargs: ".format(kwargs))
        self.write("self.request.path: {}, self.request.uri: {}".format(self.request.path, self.request.uri))
        print("request.uri : {}".format(self.request.uri))

    @gen.coroutine
    def post(self, *args, **kwargs):
        print('request.body : %s' % self.request.body)


PORT = 8000
# executor = concurrent.futures.ThreadPoolExecutor(2)

# @profile
def make_app():
    return tornado.web.Application([
        # (r"/", tornado.web.RedirectHandler, {"url": "/home"}),  # r"/" 被重定向到了"/home"
        (r"/", MainHandler),
        (r"/api", MainHandler),
        (r'/user/(.*)', MainHandler, dict(database="database")),
        (r'/([a-zA-Z0-9-]+)/123', MainHandler),
        (r'/aaa/(.*)', MainHandler),
        (r'/a/b/c', MainHandler),
        (r"/parser", MainHandler),
        ('/([a-zA-Z0-9-]+)/(.*)', MainHandler),
        ],
    )


def main():

    application = make_app()
    myserver = HTTPServer(application)
    application.listen(PORT)
    print('server is running....!!')
    tornado.ioloop.IOLoop.current().start()

if __name__ == "__main__":
    main()