#!/usr/bin/python3
# coding: utf-8

import json
import tornado.ioloop
import tornado.web

# 说明：
# 1、代码中self.request封装了所有发送过来请求的内容。
# 2、self.request.files：可以获取上传文件的所有信息。此方法获取的是一个生成器，内部是由yield实现的，因此我们在利用此方法返回的对象的时候，不能通过下标获取里面的对象，只能通过迭代的方法。
# 3、迭代出来的对象的filename：就表示上传文件的文件名。
# 4、迭代出来的对象的body：表示上传文件的内容。获取的文件内容是字节形式的。

# 注意：
# form文件上传，一定要在form表单上设置enctype的参数。enctype="multipart/form-data"。不然上传无法成功。
# 在表单中我们获取用户提交的数据，使用的是get_argument，复选框使用的是get_arguments，但是文件的不一样，文件的使用request.files。

class MainHandler(tornado.web.RequestHandler):
    def get(self):
        # self.render('index.html')  # 需要放在路径下： $PWD/template/index.html
        self.write('''
                    <html>
                    <head>
                        <meta http-equiv="Content-Type" content="text/html; charset=UTF-8"/>
                        <title>上传文件</title>
                    </head>
                    <body>
                        <form id="my_form" name="form" action="/index" method="POST"  enctype="multipart/form-data" >
                            <input name="fff" id="my_file"  type="file" />
                            <input type="submit" value="提交"  />
                        </form>
                    </body>
                    </html>
                    ''')

    def post(self, *args, **kwargs):
        file_metas = self.request.files["fff"]
        # print(file_metas)
        for meta in file_metas:
            file_name = meta['filename']
            print("上传文件名: {}".format(file_name))
            with open(file_name, 'wb') as up:
                up.write(meta['body'])
        ret = {
            "code": 0,
            "msg": '文件上传成功！'
        }
        self.write(json.dumps(ret, ensure_ascii=False))

settings = {
    'template_path': 'template',
}

application = tornado.web.Application([
    (r"/index", MainHandler),
], **settings)

if __name__ == "__main__":
    application.listen(8001)
    tornado.ioloop.IOLoop.instance().start()