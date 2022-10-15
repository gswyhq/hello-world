# -*- coding: utf-8 -*-

import urllib.request
auth_handler = urllib.request.HTTPBasicAuthHandler()
uri="https://passport.baidu.com/v2/?login&tpl=mn&u=http%3A%2F%2Fwww.baidu.com%2F"
user="用户名随便写"
passwd="密码也随便"
auth_handler.add_password(None,uri,user,passwd)
opener = urllib.request.build_opener(auth_handler)
# ...and install it globally so it can be used with urlopen.
urllib.request.install_opener(opener)
req=urllib.request.urlopen('http://tieba.baidu.com/p/3872105105')
html=req.read(1024).decode('utf-8')
print(html)

"""
HTTPBasicAuthHandler用一个叫做密码管理的对象来处理url和用户名和密码的域的映射。
如果你知道域是什么（从服务器发送的authentication 头中），那你就可以使用一个HTTPPasswordMgr。多
数情况下人们不在乎域是什么。那样使用HTTPPasswordMgrWithDefaultRealm就很方便。它允许你为一个url具体指定用户名和密码。这将会在你没有为一个特殊的域提供一个可供选择的密码锁时提供给你。
我们通过提供None作为add_password方法域的参数指出 这一点。
最高级别的url是需要authentication的第一个url。比你传递给.add_password()的url更深的url同样也会匹配。

# 创建密码管理器
password_mgr = urllib2.HTTPPasswordMgrWithDefaultRealm()
# 添加用户名和密码.
# 如果知道realm,用它代替None.
top_level_url = "http://www.163.com/"
password_mgr.add_password(None, top_level_url, username, password)
handler = urllib2.HTTPBasicAuthHandler(password_mgr)
#创建opener
opener = urllib2.build_opener(handler)
# 打开一个url
opener.open(a_url)

# 安装opener，以后urllib2.urlopen都会用它。
urllib2.install_opener(opener) 



"""

