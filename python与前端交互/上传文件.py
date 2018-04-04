#!/usr/bin/python3
# coding: utf-8

import os
import sys
import time
import json
import traceback
import tornado
import tornado.ioloop
import tornado.web
from tornado.httpserver import HTTPServer

# 前端展示代码（upload.html)：
'''
<!DOCTYPE html><html><meta charset=utf-8><title>上传手机号验证文件</title><form id="upload-form" action="/file" method="post" enctype="multipart/form-data" >
<input type="file" id="upload" name="upload" /> <br />
<input type="submit" value="上传" />
</form></html>
'''
# 在表单中还要注意，有一个action=/file，表示的是要将表单的内容提交给/file路径所对应的程序来处理。

# 前端展示代码2（上传文件及文本输入.html）：
'''
<!DOCTYPE html>
<html><meta charset=utf-8><title>上传标注的意图文件</title>
    <body>
        <form id="upload-form" action="/file" method="post" enctype="multipart/form-data" >
            <!-- 带边框的表单 -->
                <legend></legend>   <!--- 定义fieldset元素的标签 --->
                项目id：<input type="text" size="30" name="pid" ><br>
        <input type="file" id="upload" name="upload" /> <br />
        <input type="submit" value="上传" />
        </form>
    </body>
</html>
'''

SAVE_PATH = '.'

class IndexHandler(tornado.web.RequestHandler):
    def get(self):
        # 设置需要展示的表单
        self.render("communication/table.html")

class UserHandler(tornado.web.RequestHandler):
    def post(self):
        user_name = self.get_argument("username")
        user_email = self.get_argument("email")
        user_website = self.get_argument("website")
        user_language = self.get_argument("language")
        # 要引用模板网页user.html，还要向这个网页传递一些数据，例如username=user_name，含义就是，在模板中，某个地方是用username来标示得到的数据，
        # 而user_name是此方法中的一个变量，也就是对应一个数据，那么模板中的username也就对应了user_name的数据，这是通过username=user_name完成的
        # user_name = self.get_argument("username")，就能够得到index.html表单中name为"username"的元素的值，并赋给user_name变量。
        # 看HTML模板代码"communication/user.html"中，有类似{{username}}的变量，模板中用{{}}引入变量，这个变量就是在self.render()中规定的，两者变量名称一致，对应将相应的值对象引入到模板中。
        self.render("communication/user.html",username=user_name,email=user_email,website=user_website,language=user_language)

class FileUploadHandler(tornado.web.RequestHandler):
    def get(self):
        self.write('''请使用post方法上传文件''')

    def post(self):
        ret = {
            "code": 0,
            "data": '',
            "msg": "文件上传成功",
        }
        try:
            file_metas = self.request.files.get('upload', [])  # 提取表单中‘name’为‘upload’的文件元数据
            pid = self.get_argument("pid", 'unknown_project')  # 提取表单中‘name’为‘pid’的参数值
            uid = self.request.headers.get('uid', '')
            for meta in file_metas:
                file_name = meta['filename']
                print("上传文件名: {}".format(file_name))
                time_name = time.strftime('%Y-%m-%d_%H:%M:%S',time.localtime(time.time()))
                new_file_name = uid + time_name + file_name
                save_file_name = os.path.join(SAVE_PATH, new_file_name)
                print("保存的文件路径： {}".format(save_file_name))
                with open(save_file_name, 'wb') as up:
                    up.write(meta['body'])

            self.write(json.dumps(ret, ensure_ascii=False))
        except Exception as e:
            print("上传文件出错： {}".format(e))
            print("错误详情： {}".format(traceback.format_exc()))

            ret = {
                "code": 1,
                "data": '',
                "msg": "服务器错误",
            }
            self.write(json.dumps(ret, ensure_ascii=False))
        sys.stdout.flush()


def make_app():
    return tornado.web.Application([
        (r'/file', FileUploadHandler),  # 上传文件, ‘/file’与前端展示HTML中的“action”一致
        (r"/user", UserHandler),  # 接受前端提交的表单
        (r"/table3", IndexHandler),  # 浏览器打开“http://192.168.3.250:8000/table3”，即可获取对应的展示页面
        ],
        static_path= 'communication',
        static_url_prefix= '/test/',
        # debug=True
    )

PORT = 8000

def main():
    application = make_app()
    myserver = HTTPServer(application)
    myserver.listen(PORT)
    print('server is running....!!')
    tornado.ioloop.IOLoop.current().start()


if __name__ == '__main__':
    main()

