#/usr/lib/python3.5
# -*- coding: utf-8 -*-

#爬虫抓取代理服务器ip
#http://f.dataguru.cn/thread-54108-1-1.html
#############################################33
import urllib.parse
from bs4 import BeautifulSoup
import re,urllib.request,chardet
import json,time
import queue
import threading
import logging
import sys,os
path=sys.path[0]

logging.basicConfig(level=logging.DEBUG,
                format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                datefmt='%a, %d %b %Y %H:%M:%S',
                filename=os.path.join(path,'{}日志.log'.format(time.strftime("%Y%m%d%H%M"))),
                filemode='w+')

class proxy_ip():
    """从http://www.proxy360.cn/default.aspx上爬取代理IP地址及其端口号
    reptile()根据url爬取ip及端口号，
    write_txt将ip及端口号写人json文件，
    start调用多个url运行reptile。
    """
    def __init__(self,url,keyword):
        self.ip_list=self.get_proxy_ip()
        self.testUrl= url
        self.testStr = keyword

    def get_proxy_ip(self):
        """获取代理ip"""
        #从json文件加载数据
        file_name=os.path.join(path,'测试通过的代理ip',"{}抓取到的代理服务器ip.json".format(time.strftime("%Y%m%d")))
        ip_list=[]
        try:
            with open(file_name,'r',encoding='utf-8')as f:
                ip_list=json.loads(f.read())
        except FileNotFoundError:
            import 爬取代理IP
            logging.warning('重新爬取代理IP。')
            爬取代理IP.main()
            time.sleep(10)
            with open(file_name,'r',encoding='utf-8')as f:
                ip_list=json.loads(f.read())

        return ip_list

    def test_proxy(self,proxy_ip,url_timeout=1):
        """测试代理ip的可用性"""
        #testUrl= "http://www.baidu.com"
        #testStr = "030173"

        # 配置urllib.request里使用代理服务器
        cookies = urllib.request.HTTPCookieProcessor()
        #print(type(proxy_ip))
        #print(proxy_ip)
        proxy = urllib.request.ProxyHandler({'http': r'http://%s:%s' %(proxy_ip[0],proxy_ip[1])})
        opener = urllib.request.build_opener(cookies,proxy)
        opener.addheaders = [('User-agent', 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:15.0) Gecko/20100101 Firefox/15.0.1')]
        urllib.request.install_opener(opener)
        req = urllib.request.Request(self.testUrl, headers={'User-Agent': 'Proxy Tester'})
        try:
            result = urllib.request.urlopen(req,timeout=url_timeout).read().decode('utf-8')
        except Exception as e:
            logging.warning("{}测试不通过。{}".format(proxy_ip,e))
            return ''
        pos = result.find(self.testStr)
        if pos>-1:
            logging.warning("{}测试通过。".format(proxy_ip))
            return  proxy_ip
        else:
            return  ''

    def witer_ip(self,num_ip):
        """将代理ip写入json文件"""
        file_name=os.path.join(path,'测试通过的代理ip',"{}测试通过的代理服务器ip.json".format(time.strftime("%Y%m%d")))
        with open(file_name,'w+',encoding='utf-8')as f:
            f.write( json.dumps(num_ip,ensure_ascii=0))
        with open(os.path.join(path,'测试通过的代理ip','最新测试通过的代理服务器ip.json'),'w',encoding='utf-8')as f:
            f.write(json.dumps(num_ip,ensure_ascii=0))

    def reptile(self,url_list,lock,good_proxies,url_timeout=1):
        """调度"""
        while 1:
            url=url_list.get()
            test_ip=self.test_proxy(url,url_timeout)
            with lock:
                if test_ip:
                    good_proxies.append(test_ip)
            url_list.task_done()

    def start(self,threads,url_timeout=20):
        urls=self.ip_list
        url_list = queue.Queue()  # 构造一个FIFO队列先进先出。以获得代理的ip和端口 ip:ports
        lock = threading.Lock()  # 创建锁
        good_proxies = []  # 测试通过的代理
        for _ in range(threads):
            worker = threading.Thread(
                #target,线程运行的函数
                target=self.reptile,
                #args,线程的参数
                args=(
                    url_list,
                    lock,
                    good_proxies,
                    url_timeout
                    )
                )
            worker.setDaemon(True)
            #setDaemon：主线程A启动了子线程B，调用b.setDaemaon(True)，则主线程结束时，会把子线程B也杀死
            worker.start()
            #将urls中的各个数据传给reptile函数运行。

        for line in urls:
            url_list.put(line)

        # 阻塞主线程，直至代理列表为空
        url_list.join()

        self.witer_ip(good_proxies)#将爬取验证通过的ip写入文件

def main(url="http://www.baidu.com",keyword="030173"):
    start=time.clock()
    proxy=proxy_ip(url,keyword)
    proxy.start(20)
    logging.warning("测试代理ip完毕！")
    end=time.clock()
    logging.warning("测试运行时间为:{}".format(end-start))

if __name__=='__main__':
    main()


"""
import time
import re
import urllib.request
import urllib.parse
from bs4 import BeautifulSoup
import chardet

file_name="F:/python/git/抓取到的代理服务器ip.txt"
file_content = '' # 最终要写到文件里的内容
file_content += '生成时间：' + time.asctime()+'\n'
file_content +="序号\t代理IP\t端口\n"
encode={'confidence': 1.0, 'encoding': 'utf-8'}

def book_spider(url = "http://www.proxy360.cn/default.aspx"):
    global file_content,encode #global---将变量定义为全局变量。可以通过定义为全局变量，实现在函数内部改变变量值。

    #url = "http://www.proxy360.cn/default.aspx"


    #用“urllib.request.urlopen(url).read()”直径访问有些url时，会出现urllib.error.HTTPError: HTTP Error 403: Forbidden错误
    #因为如果用 urllib.request.urlopen 方式打开一个URL,服务器端只会收到一个单纯的对于该页面访问的请求,但是服务器并不知道发送这个请求使用的浏览器,操作系统,硬件平台等信息,而缺失这些信息的请求往往都是非正常的访问,例如爬虫.
    #有些网站为了防止这种非正常的访问,会验证请求信息中的UserAgent(它的信息包括硬件平台、系统软件、应用软件和用户个人偏好),如果UserAgent存在异常或者是不存在,那么这次请求将会被拒绝(如上错误信息所示)
    #如果不加上下面的这行出现会出现urllib2.HTTPError: HTTP Error 403: Forbidden错误
    #可以在请求加上头信息，伪装成浏览器访问User-Agent,具体的信息可以通过火狐的FireBug插件查询
    headers = {'User-Agent':'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:23.0) Gecko/20100101 Firefox/23.0'}
    req = urllib.request.Request(url, headers=headers)
    #con=urllib.request.urlopen(req).read().decode('utf-8')
    #网站是utf-8编码的，在这里通过 decode("UTF-8")将网站代码转换成本地系统的编码格式
    resp=urllib.request.urlopen(req)
    source_code =resp.read()

    #获取网站的编码
    encode=chardet.detect(source_code)
    #print('网站的编码是：',encode['encoding'])

    # Beautiful Soup 是用Python写的一个HTML/XML的解析器，
    #它可以很好的处理不规范标记并生成剖析树(parse tree)。
    soup = BeautifulSoup(source_code)


    count = 1
    # 得到书籍列表的soup对象
    list_soup = soup.find('td', {'style': " color:Black; width:730px;"})#通过属性进行查找
    for book_info in list_soup.findAll('div', {'class': "proxylistitem"}):
        ip = book_info.find('span', {
            'class':'tbBottomLine','style':"width:140px;"}).string.strip()#获取代理IP

        desc = book_info.find('span', {
            'class':'tbBottomLine','style':"width:50px;"}).string.strip()#获取端口

        file_content += "%4d\t%s\t%s\n" % (count,ip, desc)#返回“代理ip，端口号”
        count += 1


urls = ['http://www.proxy360.cn/Region/Brazil',
        'http://www.proxy360.cn/Region/China',
        'http://www.proxy360.cn/Region/America',
        'http://www.proxy360.cn/Region/Taiwan',
        'http://www.proxy360.cn/Region/Japan',
        'http://www.proxy360.cn/Region/Thailand',
        'http://www.proxy360.cn/Region/Vietnam',
        'http://www.proxy360.cn/Region/bahrein'
        ]
#巴西(515)、中国(504)、美国(493)、台湾(70)、日本(66)、泰国(54)、越南(25)、巴林(3)

#map(book_spider,urls)#map函数，是将urls中的各个数据传给book_spider运行。
for i in urls:
    book_spider(i)




# 将最终结果写入文件
f = open(file_name, 'w',encoding=encode['encoding'])
f.write(file_content)
f.close()

print("程序运行完毕！")

"""
