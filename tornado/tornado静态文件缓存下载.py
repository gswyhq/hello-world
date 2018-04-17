#!/usr/lib/python3
# coding = utf-8

import tornado
import tornado.ioloop
import tornado.web
from tornado.httpserver import HTTPServer
PORT = 8000

class MyFile(tornado.web.StaticFileHandler):
    def set_extra_headers(self, path):
        self.set_header("Cache-control", "no-cache")
        self.set_header("Content-Type", "text/plain; charset=utf-8")  # 若是HTML文件，用浏览器访问时，显示所有的文件内容
        # self.set_header("Content-Type", "text/html; charset=utf-8")  # 若是HTML文件，用浏览器访问时，仅仅显示body部分；

class MainHandler(tornado.web.RequestHandler):

    def get(self):
        self.write('''<html><head></head>
<body><pre style="word-wrap: break-word; white-space: pre-wrap;">#!/usr/bin/python3
# coding: utf-8

def main():
    print('测试')


if __name__ == '__main__':
    main()</pre>
</body>
</html>''')

def make_app():
    return tornado.web.Application([
        (r"/html", MainHandler),  # http://192.168.3.145:8000/html, 即可在浏览器中显示body中的内容；
        (r"/myfile/(.*)", MyFile,  {"path":"./data/"})  # 提供静态文件下载； 如浏览器打开‘http://192.168.3.145:8000/myfile/place.txt’即可访问‘./data/place.txt’文件
        ]
    )

if __name__ == "__main__":
    application = make_app()
    myserver = HTTPServer(application)
    application.listen(PORT)
    tornado.ioloop.IOLoop.current().start()
