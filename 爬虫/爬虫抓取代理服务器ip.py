# -*- coding: utf-8 -*-
#爬虫抓取代理服务器ip
#http://f.dataguru.cn/thread-54108-1-1.html
#############################################33

class proxy_ip():
    """从http://www.proxy360.cn/default.aspx上爬取代理IP地址及其端口号
    reptile()根据url爬取ip及端口号，
    write_txt将ip及端口号写人txt文件，
    start调用多个url运行reptile。
    """
    
    def reptile(self,url = "http://www.proxynova.com/proxy-server-list/"):
        import re,urllib.request,chardet
        import json
        from bs4 import BeautifulSoup
        #用“urllib.request.urlopen(url).read()”直径访问有些url时，会出现urllib.error.HTTPError: HTTP Error 403: Forbidden错误
        #因为如果用 urllib.request.urlopen 方式打开一个URL,服务器端只会收到一个单纯的对于该页面访问的请求,但是服务器并不知道发送这个请求使用的浏览器,操作系统,硬件平台等信息,而缺失这些信息的请求往往都是非正常的访问,例如爬虫.
        #有些网站为了防止这种非正常的访问,会验证请求信息中的UserAgent(它的信息包括硬件平台、系统软件、应用软件和用户个人偏好),如果UserAgent存在异常或者是不存在,那么这次请求将会被拒绝(如上错误信息所示)
        #如果不加上下面的这行出现会出现urllib2.HTTPError: HTTP Error 403: Forbidden错误
        #可以在请求加上头信息，伪装成浏览器访问User-Agent,具体的信息可以通过火狐的FireBug插件查询
        headers = {'User-Agent':'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:23.0) Gecko/20100101 Firefox/23.0'}
        req = urllib.request.Request(url, headers=headers)
        #con=urllib.request.urlopen(req).read().decode('utf-8')
        #网站是utf-8编码的，在这里通过 decode("UTF-8")将网站代码转换成本地系统的编码格式
        resp=urllib.request.urlopen(req,timeout=100)
        try:
            source_code =resp.read()
        except :
            print(url,'超时')
            return

        #获取网站的编码
        encode=chardet.detect(source_code)
        #print('网站的编码是：',encode['encoding'])

        # Beautiful Soup 是用Python写的一个HTML/XML的解析器，
        #它可以很好的处理不规范标记并生成剖析树(parse tree)。
        soup = BeautifulSoup(source_code)

        #count = 1
        file_content=''
        num_ip=[]
        # 得到书籍列表的soup对象
        list_soup = soup.find('tbody')
        for tr in list_soup.findAll('tr'):
            td= tr.findAll('td')
            if tr.find('center'):#跳过中间的广告
                break
            ip = tr.find('span', {
                'class':'row_proxy_ip'}).string.strip()
            try:
                desc = td[1].find('a').string.strip()
            except AttributeError:
                desc= td[1].string.strip()
                
            #file_content += "%4d\t%s\t%s\n" % (count,ip, desc)#返回“代理ip，端口号”
            file_content += r"http://%s:%s\n" % (ip, desc)#返回“代理ip，端口号”
            num_ip.append((ip, desc))
            #count += 1
        '''
        list_soup = soup.find('td', {'style': " color:Black; width:730px;"})#通过属性进行查找
        for book_info in list_soup.findAll('div', {'class': "proxylistitem"}):
            ip = book_info.find('span', {
                'class':'tbBottomLine','style':"width:140px;"}).string.strip()#获取代理IP
            desc = book_info.find('span', {
                'class':'tbBottomLine','style':"width:50px;"}).string.strip()#获取端口
            file_content += "%4d\t%s\t%s\n" % (count,ip, desc)#返回“代理ip，端口号”
            count += 1
        '''
        #time.strftime("%Y%m%d")，返回：'20150912'
        #time.asctime()[:11]，返回：'Sat Sep 12 
        file_name="/home/gswewf/gow69/{}抓取到的代理服务器ip.txt".format(time.strftime("%Y%m%d"))

        with open(file_name.split('.')[0]+'.json','a+',encoding='utf-8')as f:
            f.write( json.dumps(num_ip,ensure_ascii=0))


        print(encode['encoding'])
        self.write_txt(file_content,file_name,'utf-8')

    def write_txt(self,file_content='',file_name="F:/python/git/抓取到的代理服务器ip.txt",encod='utf-8'):
        # 将最终结果写入文件
        import time
        f = open(file_name, 'a+',encoding=encod)
        #r+模式打开文件时，此文件必须存在，否则就会报错，‘r’模式也如此;
        #r+模式，对文件进行write操作时，若无特殊定位，则是从文件开始写入字符，并将原字符覆盖。若写的字符，少于原文件中的字符数，则仅仅是覆盖写入部分的字符。写入字符后面的原字符保留
        #a+模式，对文件进行write操作时，是从文件的末尾，开始写入字符。
        #w+模式，对文件进行write操作时，是将新写入的字符，覆盖全部的原字符。那怕没有原文件字符数目多，原文件中超出的部分将被删除。
        #file_time = '生成时间：' + time.asctime()+'\n'
        #file_head ="序号\t代理IP\t\t端口\n"
        #f.write(file_time)
        #f.write(file_head)
        f.write(file_content)
        f.close()

    def start(self):
        urls=['http://www.proxynova.com/proxy-server-list/country-cn/',
              'http://www.proxynova.com/proxy-server-list/country-ve/',
              'http://www.proxynova.com/proxy-server-list/country-tw/',
              'http://www.proxynova.com/proxy-server-list/country-in/',
              'http://www.proxynova.com/proxy-server-list/country-br/',
              'http://www.proxynova.com/proxy-server-list/country-id/',
              'http://www.proxynova.com/proxy-server-list/country-us/',
              'http://www.proxynova.com/proxy-server-list/country-th/',
              'http://www.proxynova.com/proxy-server-list/country-ru/',
              'http://www.proxynova.com/proxy-server-list/country-ar/',
              'http://www.proxynova.com/proxy-server-list/country-hk/',
              'http://www.proxynova.com/proxy-server-list/country-de/',
              'http://www.proxynova.com/proxy-server-list/country-eg/',
              'http://www.proxynova.com/proxy-server-list/country-jp/',
              'http://www.proxynova.com/proxy-server-list/country-cl/',
              'http://www.proxynova.com/proxy-server-list/country-ma/',
              'http://www.proxynova.com/proxy-server-list/country-co/',
              'http://www.proxynova.com/proxy-server-list/country-vn/',
              'http://www.proxynova.com/proxy-server-list/country-bd/',
              'http://www.proxynova.com/proxy-server-list/country-ua/',
              'http://www.proxynova.com/proxy-server-list/country-gb/',
              'http://www.proxynova.com/proxy-server-list/country-pl/',
              'http://www.proxynova.com/proxy-server-list/country-my/',
              'http://www.proxynova.com/proxy-server-list/country-fr/'
              ]
        for i in urls:
            print('正在爬取：',i)
            self.reptile(i)
        #将urls中的各个数据传给reptile函数运行。
    
if __name__=='__main__':
    import time
    start=time.clock()
    proxy=proxy_ip()
    proxy.start()
    print("程序运行完毕！")
    end=time.clock()
    print("程序运行时间为:",(end-start))

    
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
