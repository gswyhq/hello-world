#!/usr/bin/python3
# coding: utf-8

>>> from pyftpdlib.authorizers import DummyAuthorizer
>>> from pyftpdlib.handlers import FTPHandler
>>> from pyftpdlib.servers import FTPServer
>>>
>>> authorizer = DummyAuthorizer()
>>> authorizer.add_user("user", "12345", "/home/giampaolo", perm="elradfmwMT")
>>> authorizer.add_anonymous("/home/nobody")
>>>
>>> handler = FTPHandler
>>> handler.authorizer = authorizer
>>>
>>> server = FTPServer(("127.0.0.1", 21), handler)
>>> server.serve_forever()

参考资料：http://www.jb51.net/article/110901.htm

FTP服务的主动模式和被动模式
在开始之前，先聊一下FTP的主动模式和被动模式，两者的区别 ， 用两张图来表示可能会更加清晰一些：
主动模式：

主动模式工作过程：
1.
客户端以随机非特权端口N，就是大于1024的端口，对server端21端口发起连接
2.
客户端开始监听
N + 1
端口；
3.
服务端会主动以20端口连接到客户端的N + 1
端口。
主动模式的优点：
服务端配置简单，利于服务器安全管理，服务器只需要开放21端口
主动模式的缺点：
如果客户端开启了防火墙，或客户端处于内网（NAT网关之后）， 那么服务器对客户端端口发起的连接可能会失败
被动模式：

被动模式工作过程：
1.
客户端以随机非特权端口连接服务端的21端口
2.
服务端开启一个非特权端口为被动端口，并返回给客户端
3.
客户端以非特权端口 + 1
的端口主动连接服务端的被动端口
被动模式缺点：
服务器配置管理稍显复杂，不利于安全，服务器需要开放随机高位端口以便客户端可以连接，因此大多数FTP服务软件都可以手动配置被动端口的范围
被动模式的优点：对客户端网络环境没有要求
了解了FTP之后，开始使用python来实现FTP服务
准备工作
本次使用python版本：python
3.4
.3
安装模块
pyftpdlib
?

pip3
install
pyftpdlib
创建代码文件
FtpServer.py
代码
实现简单的本地验证
?

from pyftpdlib.authorizers import DummyAuthorizer
from pyftpdlib.handlers import FTPHandler
from pyftpdlib.servers import FTPServer

# 实例化虚拟用户，这是FTP验证首要条件
authorizer = DummyAuthorizer()

# 添加用户权限和路径，括号内的参数是(用户名， 密码， 用户目录， 权限)
authorizer.add_user('user', '12345', '/home/', perm='elradfmw')

# 添加匿名用户 只需要路径
authorizer.add_anonymous('/home/huangxm')

# 初始化ftp句柄
handler = FTPHandler
handler.authorizer = authorizer

# 监听ip 和 端口,因为linux里非root用户无法使用21端口，所以我使用了2121端口
server = FTPServer(('192.168.0.108', 2121), handler)

# 开始服务
server.serve_forever()
开启服务
$python
FtpServer.py
测试一下：

输入个错误密码试试：

验证不通过，无法登录 。
但这似乎是主动模式的FTP ，如何实现被动模式呢？
通过以下代码添加被动端口：
handler.passive_ports = range(2000，2333)
完整代码：
?

from pyftpdlib.authorizers import DummyAuthorizer
from pyftpdlib.handlers import FTPHandler
from pyftpdlib.servers import FTPServer

# 实例化虚拟用户，这是FTP验证首要条件
authorizer = DummyAuthorizer()

# 添加用户权限和路径，括号内的参数是(用户名， 密码， 用户目录， 权限)
authorizer.add_user('user', '12345', '/home/', perm='elradfmw')

# 添加匿名用户 只需要路径
authorizer.add_anonymous('/home/huangxm')

# 初始化ftp句柄
handler = FTPHandler
handler.authorizer = authorizer

# 添加被动端口范围
handler.passive_ports = range(2000, 2333)

# 监听ip 和 端口
server = FTPServer(('192.168.0.108', 2121), handler)

# 开始服务
server.serve_forever()
开启服务，可以看到被动端口的信息：
?

$ python
FtpServer.py
[I 2017 - 01 - 11 15: 18:37] >> > starting
FTP
server
on
192.168
.0
.108: 2121, pid = 46296 << <
[I 2017 - 01 - 11 15: 18:37] concurrency
model: async
[I 2017 - 01 - 11 15: 18:37] masquerade(NAT)
address: None
[I 2017 - 01 - 11 15: 18:37] passive
ports: 2000->2332
FTP用户管理：
通过上面的实践，FTP服务器已经可以正常工作了，但是如果需要很多个FTP用户呢，怎么办呢？ 每个用户都写一遍吗？
其实我们可以定义一个用户文件user.py
?

# 用户名   密码    权限     目录
# root   12345   elradfmwM  /home
huangxm

elradfmwM / home
然后遍历该文件，将不以  # 开头的行加入到user_list列表中
完整代码：
?

from pyftpdlib.authorizers import DummyAuthorizer
from pyftpdlib.handlers import FTPHandler
from pyftpdlib.servers import FTPServer

def get_user(userfile):
    # 定义一个用户列表
    user_list = []
    with open(userfile) as f:
        for line in f:
            print(len(line.split()))
            if not line.startswith('#') and line:
                if len(line.split()) == 4:
                    user_list.append(line.split())
                else:
                    print("user.conf配置错误")
    return user_list

# 实例化虚拟用户，这是FTP验证首要条件
authorizer = DummyAuthorizer()

# 添加用户权限和路径，括号内的参数是(用户名， 密码， 用户目录， 权限)
# authorizer.add_user('user', '12345', '/home/', perm='elradfmw')
user_list = get_user('/home/huangxm/test_py/FtpServer/user.conf')
for user in user_list:
    name, passwd, permit, homedir = user
    try:
        authorizer.add_user(name, passwd, homedir, perm=permit)
    except Exception as e:
        print(e)

# 添加匿名用户 只需要路径
authorizer.add_anonymous('/home/huangxm')

# 初始化ftp句柄
handler = FTPHandler
handler.authorizer = authorizer

# 添加被动端口范围
handler.passive_ports = range(2000, 2333)

# 监听ip 和 端口
server = FTPServer(('192.168.0.108', 2121), handler)

# 开始服务
server.serve_forever()
到这里，FTP
服务已经完成了。
规范一下代码
首先创建conf目录，存放settings.py和user.py
目录结构(cache里面的不用管)：

setting.py
?

ip = '0.0.0.0'

port = '2121'

# 上传速度 300kb/s
max_upload = 300 * 1024

# 下载速度 300kb/s
max_download = 300 * 1024

# 最大连接数
max_cons = 150

# 最多IP数
max_per_ip = 10

# 被动端口范围，注意被动端口数量要比最大IP数多，否则可能出现无法连接的情况
passive_ports = (2000, 2200)

# 是否开启匿名访问 on|off
enable_anonymous = 'off'
# 匿名用户目录
anonymous_path = '/home/huangxm'

# 是否开启日志 on|off
enable_logging = 'off'
# 日志文件
loging_name = 'pyftp.log'

# 欢迎信息
welcome_msg = 'Welcome to my ftp'
user.py
?

# 用户名   密码    权限     目录
# root   12345   elradfmwM  /home/
huangxm

elradfmwM / home /
test

elradfmwM / home / huangxm
FtpServer.py
?

from pyftpdlib.authorizers import DummyAuthorizer
from pyftpdlib.handlers import FTPHandler, ThrottledDTPHandler
from pyftpdlib.servers import FTPServer
from conf import settings
import logging

def get_user(userfile):
    # 定义一个用户列表
    user_list = []
    with open(userfile) as f:
        for line in f:
            if not line.startswith('#') and line:
                if len(line.split()) == 4:
                    user_list.append(line.split())
                else:
                    print("user.conf配置错误")
    return user_list

def ftp_server():
    # 实例化虚拟用户，这是FTP验证首要条件
    authorizer = DummyAuthorizer()

    # 添加用户权限和路径，括号内的参数是(用户名， 密码， 用户目录， 权限)
    # authorizer.add_user('user', '12345', '/home/', perm='elradfmw')
    user_list = get_user('conf/user.py')
    for user in user_list:
        name, passwd, permit, homedir = user
        try:
            authorizer.add_user(name, passwd, homedir, perm=permit)
        except Exception as e:
            print(e)

    # 添加匿名用户 只需要路径
    if settings.enable_anonymous == 'on':
        authorizer.add_anonymous(settings.anonymous_path)

    # 下载上传速度设置
    dtp_handler = ThrottledDTPHandler
    dtp_handler.read_limit = settings.max_download
    dtp_handler.write_limit = settings.max_upload

    # 初始化ftp句柄
    handler = FTPHandler
    handler.authorizer = authorizer

    # 日志记录
    if settings.enable_logging == 'on':
        logging.basicConfig(filename=settings.loging_name, level=logging.INFO)

    # 欢迎信息
    handler.banner = settings.welcome_msg

    # 添加被动端口范围
    handler.passive_ports = range(settings.passive_ports[0], settings.passive_ports[1])

    # 监听ip 和 端口
    server = FTPServer((settings.ip, settings.port), handler)

    # 最大连接数
    server.max_cons = settings.max_cons
    server.max_cons_per_ip = settings.max_per_ip

    # 开始服务
    print('开始服务')
    server.serve_forever()

if __name__ == "__main__":
    ftp_server()
最后，说一下权限问题
读权限 ：
e
改变文件目录
l
列出文件
r
从服务器接收文件
写权限 ：
a
文件上传
d
删除文件
f
文件重命名
m
创建文件
w
写权限
M
文件传输模式（通过FTP设置文件权限 ）
M
示例：

到服务器上查看一下权限：

可以看到权限已经被修改了。

def main():
    pass

if __name__ == '__main__':
    main()