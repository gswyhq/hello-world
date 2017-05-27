#/usr/lib/python3.5
# -*- coding: utf-8 -*-

import logging
import random
import sys,os
import urllib.parse
from bs4 import BeautifulSoup
import re,urllib.request,chardet
import json,time

from multiprocessing import Pool
#from multiprocessing.dummy import Pool as ThreadPool
    
path=sys.path[0]
'''
logging.basicConfig(level=logging.DEBUG,
                format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                datefmt='%a, %d %b %Y %H:%M:%S',
                filename=os.path.join(path,'{}{}日志.log'.format(sys.argv[0],time.strftime("%Y%m%d%H%M"))),
                filemode='w+')
'''
#解析函数
def parse_html(url,html):
    """根据html解析出IP及其端口号"""
    # Beautiful Soup 是用Python写的一个HTML/XML的解析器，
    #它可以很好的处理不规范标记并生成剖析树(parse tree)。
    if not html:
        return ''

    num_ip=[]
    net_loc=urllib.parse.urlsplit(url).netloc
    if net_loc=='www.proxynova.com':
        soup = BeautifulSoup(html,"lxml")
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

            num_ip.append('{}:{}'.format(ip, desc))
    if net_loc=='www.proxy360.cn':
        # Beautiful Soup 是用Python写的一个HTML/XML的解析器，
        #它可以很好的处理不规范标记并生成剖析树(parse tree)。
        return ''

    if net_loc=="www.youdaili.net":
        soup = BeautifulSoup(html,'lxml')
        div=soup.find('div',class_='newsdetail_cont')
        for p in div.strings:
            if '@' in p:
                ip=p.strip().split('@')[0].split(':')[0].strip()
                desc=p.strip().split('@')[0].split(':')[1].strip()
                num_ip.append('{}:{}'.format(ip, desc))

    if net_loc=="www.proxylists.net":
        return ''
    if net_loc=="proxylist.sakura.ne.jp":
        return ''
    if net_loc=="www.my-proxy.com":
        return ''
    if net_loc=="www.cybersyndrome.net":
        return ''
    if net_loc=="cn-proxy.com":
        return ''
    if net_loc=="www.youdaili.cn":
        return ''
    if net_loc=="www.itmop.com":
        return ''
    if net_loc=="pachong.org":
        return ''
    if net_loc=="www.cz88.net":
        return ''
    if net_loc=="www.kuaidaili.com":
        return ''
    if net_loc=="www.xicidaili.com":
        soup = BeautifulSoup(html,'lxml')
        # 得到列表的soup对象
        list_soup = soup.find('table', {'id': "ip_list"})#通过属性进行查找
        for tr in list_soup.findAll('tr'):
            td=tr.findAll('td')
            #print(tr)
            if len(td)>3 and 'HTTP'in tr.get_text():
                ip=td[1].string.strip()
                desc=td[2].string.strip()
                num_ip.append('{}:{}'.format(ip, desc))

    return num_ip
     
def test_proxy(proxy_ip,url_timeout=1):
    """测试代理ip的可用性"""
    testUrl= "http://www.baidu.com"
    testStr = "030173"

    # 配置urllib.request里使用代理服务器
    cookies = urllib.request.HTTPCookieProcessor()
    #print(type(proxy_ip))
    #print(proxy_ip)
    proxy = urllib.request.ProxyHandler({'http': r'http://{}'.format(proxy_ip)})
    opener = urllib.request.build_opener(cookies,proxy)
    opener.addheaders = [('User-agent', 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:15.0) Gecko/20100101 Firefox/15.0.1')]
    urllib.request.install_opener(opener)
    req = urllib.request.Request(testUrl, headers={'User-Agent': 'Proxy Tester'})
    try:
        result = urllib.request.urlopen(req,timeout=url_timeout).read().decode('utf-8')
    except Exception as e:
        logging.warning("{}测试不通过。{}".format(proxy_ip,e))
        return ''
    pos = result.find(testStr)
    if pos>-1:
        #print(result)
        return  proxy_ip
    else:
        return  ''

def get_html(url,ip_list):
    #用“urllib.request.urlopen(url).read()”直径访问有些url时，会出现urllib.error.HTTPError: HTTP Error 403: Forbidden错误
    #因为如果用 urllib.request.urlopen 方式打开一个URL,服务器端只会收到一个单纯的对于该页面访问的请求,但是服务器并不知道发送这个请求使用的浏览器,操作系统,硬件平台等信息,而缺失这些信息的请求往往都是非正常的访问,例如爬虫.
    #有些网站为了防止这种非正常的访问,会验证请求信息中的UserAgent(它的信息包括硬件平台、系统软件、应用软件和用户个人偏好),如果UserAgent存在异常或者是不存在,那么这次请求将会被拒绝(如上错误信息所示)
    #如果不加上下面的这行出现会出现urllib2.HTTPError: HTTP Error 403: Forbidden错误
    #可以在请求加上头信息，伪装成浏览器访问User-Agent,具体的信息可以通过火狐的FireBug插件查询

    for i in range(10):
        proxy_ip=random.choice(ip_list)
        cookies = urllib.request.HTTPCookieProcessor()
        proxy = urllib.request.ProxyHandler({'http': r'http://{}' .format(proxy_ip)})
        opener = urllib.request.build_opener(cookies,proxy)
        opener.addheaders = [('User-agent', 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:15.0) Gecko/20100101 Firefox/15.0.1')]
        urllib.request.install_opener(opener)
        req = urllib.request.Request(url, headers={'User-Agent': 'Proxy Tester'})
        #con=urllib.request.urlopen(req).read().decode('utf-8')
        #网站是utf-8编码的，在这里通过 decode("UTF-8")将网站代码转换成本地系统的编码格式
        try:
            resp=urllib.request.urlopen(req,timeout=100)

            source_code =resp.read()
            #获取网站的编码
            encode=chardet.detect(source_code)
            #print('网站的编码是：',encode['encoding'])
            if not encode['encoding']:
                import  zlib
                data=zlib.decompress(source_code, 16+zlib.MAX_WBITS);#解决返回网页被压缩了的问题
                html=data.decode('utf-8')
                return parse_html(url,html)
            try:
                html=source_code.decode(encode['encoding'])
                return parse_html(url,html)
            except UnicodeDecodeError:
                html= source_code.decode('gb18030')
                return parse_html(url,html)
        except Exception as e :
            logging.warning('{}出错{}'.format(url,e))
            #return
    return ''

def jiexi_youdaili(ip_list=['42.200.35.27:3128', '218.76.84.201:3128', '190.54.112.159:3128'],url='http://www.youdaili.net/Daili/'):
    html=get_html(url,ip_list)
    # Beautiful Soup 是用Python写的一个HTML/XML的解析器，
    #它可以很好的处理不规范标记并生成剖析树(parse tree)。
    #若是主页，则解析出href属性的网页
    urls=[]
    soup = BeautifulSoup(html,'lxml')
    for img in soup.findAll('img',alt='热门图标'):
        href=img.find_parent('a',target='_blank')
        if href:
            urls.append(href['href'])
    '''
    for book_info in soup.findAll('li'):
        atarget=book_info.find('a',{'target':"_blank"})
        if atarget:
            if 'shezhi' in atarget['href']:
                continue
            if 'Zhishi' in atarget['href']:
                continue
            if 'news' in atarget['href']:
                continue
            if 'soft' in atarget['href']:
                continue
            urls.append(atarget['href'])
    '''
    return urls

def get_ip(ip_list):
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
              'http://www.proxynova.com/proxy-server-list/country-fr/',
              'http://www.xicidaili.com/'
              ]  
    urls.extend(jiexi_youdaili(ip_list=ip_list))
    with Pool()as pool:
        data=pool.starmap(get_html,zip(urls,[[ip_list]]*len(urls)))
    
def main():
    
    with open(os.path.join(path,'测试通过的代理ip','最新测试通过的代理服务器ip.json'),'r',encoding='utf-8')as f:
        ip_list=json.load(f)
    ips=get_ip(ip_list)
    with open(os.path.join(path,'测试通过的代理ip','最新爬取的代理服务器ip.json'),'w',encoding='utf8')as f:
        json.dump(ips,f,ensure_ascii=0)
    
    


if __name__ == "__main__":
    main()
