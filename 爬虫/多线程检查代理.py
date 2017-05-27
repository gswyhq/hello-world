# -*- coding: utf-8 -*-
#多线程检查代理

#http://codereview.stackexchange.com/questions/97722/multithreaded-proxy-checker-in-python

""" 多线程代理检查

给出一个包含代理的文件，每行，以IP形式：端口，将尝试
建立通过每个代理提供的URL连接。持续时间
连接尝试由在超时值传递管辖。此外，
旋转关闭一些守护线程的使用传递到加快处理速度
螺纹参数。该测试通过代理被写入到一个文件 results.txt

使用方法:

    goodproxy.py [-h] -file FILE -url URL [-timeout TIMEOUT] [-threads THREADS]

参数:

    -file    -- 包含IP列表文件名：每行一个代理ip及端口，如：http://195.40.6.43:8080
    -url     -- 测试连接的url
    -timeout -- 连接超时 (默认 1.0秒)
    -threads -- 线程数设置 (默认 16个)

功能:

    get_proxy_list_size  -- 返回代理 holdingQueue的大小
    test_proxy            -- 经由代理连接到url
    main                 -- 主线程，将结果写入文件

"""
import argparse #argparse模块的作用是用于解析命令行参数
import queue
import socket
import sys
import threading
import time
import urllib.request
import http.client


def get_proxy_list_size(proxy_list):
    """ 返回当前队列（ip:端口， ip:ports）大小。 """

    return proxy_list.qsize()#qsize() 返回队列的大小


def test_proxy(url, url_timeout, proxy_list, lock, good_proxies, bad_proxies):
    """ 尝试建立通过代理到传入URL的连接。

    此功能不断，而用在守护线程和意志环
    等待在proxy_list可用的代理服务器。一旦proxy_list包含
    代理，该功能将提取的代理。这个动作自动
    锁定队列中，直到该线程完成它。构建一个urllib.request里
    首战与代理进行配置。尝试打开URL，并如果
    successsful然后保存好代理到good_proxies列表。如果
    抛出异常，不良代理写入到bodproxies列表。呼叫
    到task_done（）结尾解锁队列，以便进一步处理。

    """

    while True:

        # 采取从代理列表队列中的项目;得到（）自动锁
        # 队列此线程使用
        #队列对象的get()方法从队头删除并返回一个项目。可选参数为block，默认为True。
        #如果队列为空且block为True，
        #get()就使调用线程暂停，直至有项目可用。如果队列为空且block为False，队列将引发Empty异常。
        proxy_ip = proxy_list.get()

        # 配置urllib.request里使用代理服务器
        proxy = urllib.request.ProxyHandler({'http': proxy_ip})
        opener = urllib.request.build_opener(proxy)
        urllib.request.install_opener(opener)

        # 一些网站阻止频繁的查询从通用头
        request = urllib.request.Request(
            url, headers={'User-Agent': 'Proxy Tester'})

        try:
            # 尝试建立连接
            urllib.request.urlopen(request, timeout=float(url_timeout))

            # 如果一切顺利的好代理保存到列表
            with lock:
                good_proxies.append(proxy_ip)
                #with open(r"F:\python\git\代理ip20150912.txt", 'a+') as result_file:
                    #result_file.write(proxy_ip+'\n')

        except (urllib.request.URLError,
                urllib.request.HTTPError,
                http.client.BadStatusLine,#服务器响应的状态码不能被理解时引发
                socket.error):

            # 处理错误连接 (timeouts, refused
            # connections, HTTPError, URLError, 等)
            with lock:
                bad_proxies.append(proxy_ip)

        finally:
            proxy_list.task_done()
            # 每一个get()调用得到一个任务，接下来的task_done()调用告诉队列该任务已经处理完毕。
            #在主线程中使用了 queue.join()，导致主线程阻塞。queue.task_done() 表示完成一个 task，
            #并递减没有完成的队列数，
            #当队列全部完成时候，没有task可执行，因此需要发送一个信号，通知被阻塞的主线程，继续运行。

def main(argv):
    """ 主要功能

    使用argparse处理输入参数。文件和URL是必需的，而
    超时和线程值是可选的。使用线程创建
    守护线程的每个监控队列可用号码
    代理进行测试。一旦队列开始填充，等待守护进程
    线程将开始拿起了代理和测试它们。成功
    结果写出到RESULTS.TXT文件。

    """

    proxy_list = queue.Queue()  # 构造一个FIFO队列先进先出。以获得代理的ip和端口 ip:ports
    lock = threading.Lock()  # 创建锁 good_proxies, bad_proxies lists
    good_proxies = []  # 通过测试的代理
    bad_proxies = []  # 测试失败的代理

    # 参数输入过程
    parser = argparse.ArgumentParser(description='检测代理')
    #创建一个命令行参数解析对象
    #方法ArgumentParser(prog=None, usage=None,description=None,
    #epilog=None, parents=[],formatter_class=argparse.HelpFormatter,
    #prefix_chars='-',fromfile_prefix_chars=None,
    #argument_default=None,conflict_handler='error', add_help=True)

    #向命令行参数解析对象中添加命令行参数和选项，每一个add_argument方法对应一个参数或选项；
    #方法add_argument(name or flags...[, action][, nargs][, const][, default][, type]
    #[, choices][, required][, help][, metavar][, dest])
    #其中：
    #name or flags：命令行参数名或者选项，如上面的address或者-p,--port.其中命令行参数如果没给定，
    #且没有设置defualt，则出错。但是如果是选项的话，则设置为None
    #nargs：命令行参数的个数，一般使用通配符表示，其中，'?'表示只用一个，'*'表示0到多个，'+'表示至少一个
    #default：默认值
    #type：参数的类型，默认是字符串string类型，还有float、int等类型
    #help：和ArgumentParser方法中的参数作用相似，出现的场合也一致
    parser.add_argument(
        '-file', help='代理列表的文本文件，每行一个代理（如：http://195.40.6.43:8080）',
        required=True)
    parser.add_argument(
        '-url', help='尝试连接的url', required=True)
    parser.add_argument(
        '-timeout',
        type=float, help='超时设置，默认为1秒。', default=1)
    parser.add_argument(
        '-threads', type=int, help='线程数设置，默认为16.',
        default=16)

    #调用parse_args()方法进行解析命令行参数
    args = parser.parse_args(argv)

    # 设置 ^._.^
    '''
    threading.Thread类的初始化函数原型：
    def __init__(self, group=None, target=None, name=None, args=(), kwargs={})
    参数group是预留的，用于将来扩展；
    参数target是一个可调用对象（也称为活动[activity]），在线程启动后执行；
    参数name是线程的名字。默认值为“Thread-N“，N是一个数字。
    参数args和kwargs分别表示调用target时的参数列表和关键字参数。
    '''
    for _ in range(args.threads):
        worker = threading.Thread(
            target=test_proxy,
            args=(
                args.url,
                args.timeout,
                proxy_list,
                lock,
                good_proxies,
                bad_proxies
                )
            )
        worker.setDaemon(True)
        #setDaemon：主线程A启动了子线程B，调用b.setDaemaon(True)，则主线程结束时，会把子线程B也杀死
        worker.start()

    start = time.time()

    # 从代理文件加载代理列表
    with open(args.file) as proxyfile:
        for line in proxyfile:
            proxy_list.put(line.strip())
    #put()方法在队尾插入一个项目。put()有两个参数，第一个item为必需的，为插入项目的值；
    #第二个block为可选参数，默认为1。如果队列当前为空且block为1，put()方法就使调用线程暂停,
    #直到空出一个数据单元。如果block为0，put方法将引发Full异常。

    # 阻塞主线程，直至代理列表为空
    proxy_list.join()

    # 将代理列表写入文件
    with open(r"{}\可访问{}的代理{}.txt".format('F:\python\git',args.url.split('/')[-1],time.strftime("%Y%m%d")), 'w') as result_file:
        result_file.write('\n'.join(good_proxies))

    # 一些指标
    print("运行时间: {0:.2f}s".format(time.time() - start))


if __name__ == "__main__":
    #main(sys.argv[1:])
    main(sys.argv[1:])
    print('程序运行完毕')
    #在f:\python\git\文件夹中打开cmd，输入：
    # python 多线程检查代理.py -file 0905抓取到的代理服务器ip.txt -url http://www.baidu.com
    
