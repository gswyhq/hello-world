#!/usr/bin/env python
#coding:utf-8

import os.path

import tornado.httpserver
import tornado.ioloop
import tornado.options
import tornado.web

from tornado.options import define, options
define("port", default=8000, help="run on the given port", type=int)

'''
做为tornado驱动的网站，首先要能够把前面的index.html显示出来，这个一般用get方法，显示的样式就按照上面的样子显示。

用户填写信息之后，点击按钮提交。注意观察上面的代码表单中，设定了post方法，所以，在python程序中，应该有一个post方法专门来接收所提交的数据，然后把提交的数据在另外一个网页显示出来。

在表单中还要注意，有一个action=/user，表示的是要将表单的内容提交给/user路径所对应的程序来处理。这里需要说明的是，在网站中，数据提交和显示，路径是非常重要的。
'''
class IndexHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("index.html")

class UserHandler(tornado.web.RequestHandler):
    def post(self):
        user_name = self.get_argument("username")
        user_email = self.get_argument("email")
        user_website = self.get_argument("website")
        user_language = self.get_argument("language")
        self.render("user.html",username=user_name,email=user_email,website=user_website,language=user_language)
        # 引用模板网页user.html，还要向这个网页传递一些数据，例如username = user_name，含义就是，在模板中，某个地方是用username来标示得到的数据，而user_name是此方法中的一个变量，也就是对应一个数据，那么模板中的username也就对应了user_name的数据，这是通过username = user_name完成的（说的有点像外语了）。后面的变量同理。
        #
        # 那么，user_name的数据是哪里来的呢？就是在index.html页面的表单中提交上来的。注意观察路径的设置，r"/user", UserHandler，也就是在form中的action = '/user'，就是要将数据提交给UserHandler处理，并且是通过post方法。所以，在UserHandler类中，有post()
        # 方法来处理这个问题。通过self.get_argument()
        # 来接收前端提交过来的数据，接收方法就是，self.get_argument()
        # 的参数与index.html表单form中的各项的name值相同，就会得到相应的数据。例如user_name = self.get_argument("username")，就能够得到index.html表单中name为
        # "username"
        # 的元素的值，并赋给user_name变量。

        # 看HTML模板代码中，有类似{{username}}的变量，模板中用{{}}引入变量，这个变量就是在self.render()中规定的，两者变量名称一致，对应将相应的值对象引入到模板中。

handlers = [
    (r"/", IndexHandler),
    (r"/user", UserHandler)
]

template_path = os.path.join(os.path.dirname(__file__),"template")  # 保障‘self.render('index.html')’有效果

if __name__ == "__main__":
    tornado.options.parse_command_line()
    app = tornado.web.Application(handlers, template_path)
    http_server = tornado.httpserver.HTTPServer(app)
    http_server.listen(options.port)
    tornado.ioloop.IOLoop.instance().start()

