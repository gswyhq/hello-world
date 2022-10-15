# -*- coding: utf-8 -*-
#检验代理ip
import time,json
import urllib.request
import http.client
import socket
class sim_ProxyCheck():
    def __init__(self):
        self.timeout = 10
        self.testUrl = "http://www.baidu.com/"
        self.testStr = "030173"
        
    def checkProxy(self):
        cookies = urllib.request.HTTPCookieProcessor()
        #从json文件加载数据
        with open("/home/gswewf/gow69/{}抓取到的代理服务器ip.json".format(time.strftime("%Y%m%d")),'r',encoding='utf-8')as f:
            txt_ip=json.loads(f.read())
        print(txt_ip)
        proxyHandler = urllib.request.ProxyHandler({"http" : r'http://%s:%s' %('218.249.3.133','8080')})
        opener = urllib.request.build_opener(cookies,proxyHandler)
        opener.addheaders = [('User-agent', 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:15.0) Gecko/20100101 Firefox/15.0.1')]

        #socket.setdefaulttimeout(self.timeout)#访问超时设置
        req = urllib.request.Request(self.testUrl)
        result = urllib.request.urlopen(req).read().decode('utf-8')
        pos = result.find(self.testStr)
        if (pos > -1):
            print("检测通过。")
        else:
            print ("检测代理IP出错")



def test_proxy(f,url_timeout=1):
    
    url = "http://www.baidu.com"
    proxy_list=[]
    good_proxies=[]
    #读取代理列表
    with open(f,'r',encoding='gbk') as proxyfile:
        for line in proxyfile.readlines():
            proxy_list.append(line.strip())

    for proxy_ip in proxy_list:
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

        except (urllib.request.URLError,
                urllib.request.HTTPError,
                http.client.BadStatusLine,#服务器响应的状态码不能被理解时引发
                socket.error):
            continue
        good_proxies.append(proxy_ip)

    #将可用的代理ip 写入文件
    with open(r"{}\可访问{}的代理{}.txt".format('F:\python\git',url.split('/')[-1],time.strftime("%Y%m%d")), 'w') as result_file:
        for ip in good_proxies:
            print(ip,file=result_file)

if __name__ == "__main__":
    start=time.clock()

    #方法一：
    #p=sim_ProxyCheck()
    #p.checkProxy()

    #方法二：
    test_proxy(r'F:\python\git\0905抓取到的代理服务器ip.txt')
    #"""
    end=time.clock()
    print("程序运行时间为:",(end-start))

    
    """
>>> def proxy(url):
	import urllib.request
	proxy_support = urllib.request.ProxyHandler({'sock5': 'localhost:1080'})
	opener = urllib.request.build_opener(proxy_support)
	urllib.request.install_opener(opener)
	a = urllib.request.urlopen(url).read().decode("utf8")
	print(a)

	
>>> url="http://www.baidu.com"
>>> proxy(url)
    """
