#!/usr/bin/python3
# coding: utf-8

import tornado.ioloop
import tornado.web


class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.write("Hello, world")


class DomainHandler(tornado.web.RequestHandler):
    def get(self):
        self.write("Hello, a.com")


application = tornado.web.Application([
    (r"/", MainHandler),
])

application.add_handlers(r"^a\.com$", [
    (r"/", DomainHandler),
])

# application.add_handlers(r"^(www\.)?a\.com$", [(r"/", DomainHandler),])

def main():
    application.listen(8888)
    tornado.ioloop.IOLoop.instance().start()


if __name__ == '__main__':
    main()

# http://localhost:8888/
# http: //www.a.com: 8888/
# http: //www.a.com: 8888/

# 一个网站要正常常访问，需要满足2个条件：
# 1.
# 域名解析到空间的IP地址。
# 2.
# 空间绑定了域名。
#
# 通过你的描述，应该是你没有做第二项，你的域名能能访问到服务器，但服务器没有绑定域名，所以不知道要指向到这个服务器的哪个空间里去，因此出现了你看到的那些提示。
#
# 解决办法：
# 在你的空间管理平台中，有一项“绑定域名”，只要添加域名绑定即可。
# 如果你没有域名管理平台的操作权限，可联系你的空间服务商，让他们帮你完成。这个操作很简单，服务商二分钟就可帮你完成。
#
# 怎把域名绑定在虚拟主机上
# 你申请主机的时候会让填域名吧？如果没有填，如下：
# 第一步，解析，进入域名管理面板，把A记录指向你申请的虚拟主机的IP地址。
# 第二步，备案（国外主机不用备），联系你申请空间的服务商，他会告诉你怎么做。
# 第三步，绑定，进入主机管理面板，有增加域名的选项，增加绑定就可以了。
# 如果以上感觉不明白，那你可以采用个极简单的方法，问你的虚拟主机服务商的客服，要来你的IP，再联系你的域名服务商，把这个IP给他，让他给你解析到这个IP地址。然后联系你的虚拟主机服务商，让他帮你绑定。