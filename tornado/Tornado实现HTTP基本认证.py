#!/usr/bin/python3
# coding: utf-8

# HTTP 基本认证的过程：
#
# 1．客户端向服务器请求数据，请求的内容可能是一个网页或者是一个其它的MIME类型，此时，假设客户端尚未被验证，则客户端提供如下请求至服务器:
#
# Get /index.html HTTP/1.0
# Host:www.google.com
# 2． 服务器向客户端发送验证请求代码401,服务器返回的数据大抵如下：
#
# HTTP/1.0 401 Unauthorised
# Server: SokEvo/1.0
# WWW-Authenticate: Basic realm=”google.com”
# Content-Type: text/html
# Content-Length: xxx
# 3． 当符合 http1.0 或 1.1 规范的客户端（如IE，FIREFOX）收到401返回值时，将自动弹出一个登录窗口，要求用户输入用户名和密码。
# 4． 用户输入用户名和密码后，将用户名及密码以 BASE64 加密方式加密，并将密文放入前一条请求信息中，则客户端发送的第一条请求信息则变成如下内容：
#
# Get /index.html HTTP/1.0
# Host:www.google.com
# Authorization: Basic xxxxxxxxxxxxxxxxxxxxxxxxxxxx
# # 注：xxxx….表示加密后的用户名及密码。
# 5． 服务器收到上述请求信息后，将 Authorization 字段后的用户信息取出、解密，将解密后的用户名及密码与用户数据库进行比较验证，如用户名及密码正确，服务器则根据请求，将所请求资源发送给客户端：

import tornado.web
import tornado.ioloop
from tornado.httpserver import HTTPServer
import base64

class HTTPMainHandler(tornado.web.RequestHandler):
    def get(self):
        auth_header = self.request.headers.get('Authorization', None)
        print('auth_header', auth_header)
        if auth_header is not None:
            auth_mode, auth_base64 = auth_header.split(' ', 1)
            auth_username, auth_password = base64.b64decode(auth_base64).decode('utf8').split(':', 1)
            print(auth_mode, auth_base64, auth_username, auth_password)
            if auth_username == 'hello' and auth_password == 'world':
                self.write('登录成功！')
            else:
                # self.set_header('WWW-Authenticate', 'Basic realm="%s"' % 'hello')  # 若不注释，则可保证登录失败后，继续弹出登录框
                self.write('登录失败！')
                self.set_status(401)  # 保证刷新界面之后，能再次弹出登录框
        else:
            '''
            HTTP/1.1 401 Unauthorized
            WWW-Authenticate: Basic realm="renjie"
            '''
            self.set_status(401)
            self.set_header('WWW-Authenticate', 'Basic realm="%s"' % 'hello')
            # print('WWW-Authenticate')

def make_app():
    return tornado.web.Application([
        (r'/http', HTTPMainHandler),
        ],
        # debug=False,
        # cookie_secret='abcde',
        # expires_days=None,
    )

def main():
    application = make_app()
    myserver = HTTPServer(application)
    application.listen(8800)
    print('server is running....!!')
    tornado.ioloop.IOLoop.current().start()



if __name__ == '__main__':
    main()
    
# 浏览器登录：http://192.168.3.145:8800/http
# 或者用户名密码登录：
# gswyhq@gswyhq-PC:~/ner_es$ curl http://192.168.3.145:8800/http --u hello:world
# gswyhq@gswyhq-PC:~/ner_es$ curl http://192.168.3.145:8800/http -u hello:world
#
# import requests
# r = requests.get('http://192.168.3.145:8800/http', auth=('hello', 'world'))
# r.text
# Out[31]: '登录成功！'
