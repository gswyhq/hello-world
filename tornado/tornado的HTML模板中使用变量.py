#!/usr/bin/python3
# coding: utf-8

import json
import tornado.ioloop
import tornado.web

# Tornado 模板其实就是 HTML 文件（也可以是任何文本格式的文件），其中包含了 Python 控制结构和表达式，这些控制结构和表达式需要放在规定的格式标记符(markup)中：
'''<html>
   <head>
      <title>{{ title }}</title>
   </head>
   <body>
     <ul>
       {% for item in items %}
         <li>{{ escape(item) }}</li>
       {% end %}
     </ul>
   </body>
 </html>
 '''

# 如果你把上面的代码命名为 "template.html"，保存在 Python 代码的同一目录中，你就可以 这样来渲染它：

class MainHandler(tornado.web.RequestHandler):
    def get(self):
        items = ["Item 1", "Item 2", "Item 3"]
        self.render("template.html", title="My title", items=items)

application = tornado.web.Application([
    (r"/index", MainHandler),
])

if __name__ == "__main__":
    application.listen(8001)
    tornado.ioloop.IOLoop.instance().start()

# Tornado 的模板支持“控制语句”和“表达语句”，控制语句是使用 {% 和 %} 包起来的 例如 {% if len(items) > 2 %}。表达语句是使用 {{ 和 }} 包起来的，例如 {{ items[0] }}。


# 一、关于模板中符合的使用
#
# 1、{{}}双大括号，内容可以是任何的python表达式(变量常见)
# 2、{%%}模板中的控制语句放在{%%}中
# 二、关于{{}}的使用
#
# 1、传递变量
#
# class IndexHandler(tornado.web.RequestHandler):
#     def get(self):
#         name = u'张三'
#         age = 20
#         self.render("template-demo.html",name=name,age=age)
#
# <p>{{ name }}</p>
# <p>{{ age }}</p>
# 1
# 2
# 2、一般不会一个一个参数传递的，直接传递一个对象
#
# class IndexHandler(tornado.web.RequestHandler):
#     def get(self):
#         resultDate = {
#             'name':u'张三',
#             'age':20
#         }
#         self.render("template-demo.html",**resultDate)
#
# 3、在{{}}中使用表达式
#
# <p>{{ 1+2 }}</p>
# 1
# 4、在{{}}使用函数
#
# class IndexHandler(tornado.web.RequestHandler):
#     def foo(self):
#         return u'我是函数'
#
#     def get(self):
#         resultDate = {
#             'name':u'张三',
#             'age':20,
#             'foo':self.foo
#         }
#         self.render("template-demo.html",**resultDate)
#
# <p>{{ name }}</p>
# <p>{{ age }}</p>
# <p>{{ 1+2 }}</p>
# <p>{{ foo() }}</p>
# 1
# 2
# 3
# 4
# 三、关于{% %}的使用
#
# 1、模板的控制语句(注意要结束语句)
#
# {% if age > 20 %}
# ...
# {% elif %}
# ...
# {% else %}
# ...
# {% end %}
#
# 2、模板中使用for语句
#
# {% for item in list1 %}
#
# {% end %}
#
# class IndexHandler(tornado.web.RequestHandler):
#     def get(self):
#         resultDate = {
#             'names':[u'张三',u'李四',u'王五'],
#             'urls':[
#                 ('https://www.hao123.com/','hao123'),
#                 ('http://www.sina.com.cn/','新浪')
#             ]
#         }
#         self.render("template-demo.html",**resultDate)
#
# {% for item in names %}
#     <p>{{ item }}</p>
# {% end %}
#
# {% for url in urls %}
#     <p><a href="{{ url[0] }}">{{ url[1] }}</a> </p>
# {% end %}
#
# 四、使用static_url加载静态文件
#
# <link rel="stylesheet" href="{{ static_url('xxx/xx.css') }}">

# 来源： https://blog.csdn.net/kuangshp128/article/details/73937294
