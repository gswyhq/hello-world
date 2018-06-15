#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import tornado.ioloop
import tornado.web
import os

class FileDownloadHandler(tornado.web.RequestHandler):
    def get(self):
        # Content-Type这里我写的时候是固定的了，也可以根据实际情况传值进来
        self.set_header('Content-Type', 'application/octet-stream;charset=utf-8')
        filename = "自定义下载的文件名.json".encode('utf-8')
        self.set_header('Content-Disposition', b'attachment; filename=' + filename)
        with open("./output/同义词.json", "rb") as f:
            self.set_header('Content-Type', 'application/octet-stream')
            self.write(f.read())
        logger.info("请求下载文件")
        # self.finish()
        sys.stdout.flush()


# @profile
def make_app():
    return tornado.web.Application([
        (r'/download', FileDownloadHandler),  # 下载文件
        ],
    )

# 浏览器打开： http://192.168.3.145:8000/download， 即可进行下载；

def main():
    application = make_app()
    myserver = HTTPServer(application)
    application.listen(8000)
    logger.info('server is running....!!')
    tornado.ioloop.IOLoop.current().start()

if __name__ == "__main__":
    main()