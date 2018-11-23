tornado 启动时加上 SSL 选项
复制证书文件到 tornado server 目录下(可选)

修改测试服务器代码 test.py


import tornado.ioloop
import tornado.web
import os

class TestGetHandler(tornado.web.RequestHandler):
    def get(self):
        self.write("Hello, World!")

def make_app():
    return tornado.web.Application([
        (r"/", TestGetHandler),
    ])

if __name__ == "__main__":
    application = make_app()
    http_server = tornado.httpserver.HTTPServer(application, ssl_options={
           "certfile": os.path.join(os.path.abspath("."), "server.crt"),
           "keyfile": os.path.join(os.path.abspath("."), "server.key"),
    })
    http_server.listen(443)
    tornado.ioloop.IOLoop.instance().start()
由于端口号小于1000, 因此需要使用 su 权限的用户运行脚本
$ sudo python test.py

客户端访问
浏览器

输入https://localhost直接访问

CURL

添加 -k 选项忽略 SSL 验证, 如下:

curl -k https://localhost

requests

添加verify=False选项, 如下:

requests.get(URL, verify=False)

https://www.cnblogs.com/ityoung/p/8296088.html


文件 rootCA-key.pem  rootCA.pem 的生成：
1、下载预编译好的二进制文件：https://github.com/FiloSottile/mkcert/releases/download/v1.0.0/mkcert-v1.0.0-linux-amd64
2、生成*.pem文件： ~$ ./mkcert-v1.0.0-linux-amd64

