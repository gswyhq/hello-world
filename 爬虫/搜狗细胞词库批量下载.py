#! /usr/lib/python3
# -*- coding: utf-8 -*-

import  urllib.parse,http.cookiejar,urllib.request
from bs4 import BeautifulSoup
#from pprint import pprint
import sys,chardet,json
import  logging,random,os,time,socket

'''
urls=['http://pinyin.sogou.com/dict/cate/index/167',
 'http://pinyin.sogou.com/dict/cate/index/1',
 'http://pinyin.sogou.com/dict/cate/index/76',
 'http://pinyin.sogou.com/dict/cate/index/96',
 'http://pinyin.sogou.com/dict/cate/index/127',
 'http://pinyin.sogou.com/dict/cate/index/132',
 'http://pinyin.sogou.com/dict/cate/index/436',
 'http://pinyin.sogou.com/dict/cate/index/154',
 'http://pinyin.sogou.com/dict/cate/index/389',
 'http://pinyin.sogou.com/dict/cate/index/367',
 'http://pinyin.sogou.com/dict/cate/index/31',
 'http://pinyin.sogou.com/dict/cate/index/403']


Accept:text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8
Accept-Encoding:gzip, deflate, sdch
Accept-Language:zh-CN,zh;q=0.8
Connection:keep-alive
Cookie:CXID=3A255536B9444AEBD5B4FE213B463F87; SUID=09E10EB7546C860A56DFF78B000D86B4; IPLOC=CN4403; ssuid=8295395525; SUV=1458207705399002; SMYUV=1459821161803055; sogoupy_iweb_cnzz_eid1_0_1=739472-1459821833-http%253A%252F%252Fpinyin.sogou.com%252Fhelp.php%253Flist%253D6%2526q%253D3; redref=http://pinyin.sogou.com/func/; SEID=000000004658860A30E8E4C5000A32B3; sogoupy_iweb_ntime1_0_1=1459911109; ad=Tyllllllll2QT@WalllllVt9u9DlllllHZEPjyllllGlllllxOxlw@@@@@@@@@@@
Host:download.pinyin.sogou.com
Referer:http://pinyin.sogou.com/dict/cate/index/128
Upgrade-Insecure-Requests:1
User-Agent:Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/49.0.2623.75 Safari/537.36


def makeMyOpener(Referer="",proxy_ip=[]):
    head = {
    "Accept":"text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Encoding":"gzip, deflate, sdch",
    "Accept-Language":"zh-CN,zh;q=0.8",
    #"Connection":"keep-alive",
    "Host":"download.pinyin.sogou.com",
    "Upgrade-Insecure-Requests":1,
    "User-Agent":"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/49.0.2623.75 Safari/537.36"
    }
    if Referer:
        #Referer="http://pinyin.sogou.com/dict/cate/index/128"
        head["Referer"]=Referer
    cj = http.cookiejar.CookieJar()
    cookies = urllib.request.HTTPCookieProcessor(cj)
    if proxy_ip:
        # proxy_ip=['176.31.239.33', '8118']
        proxy = urllib.request.ProxyHandler({'http': r'http://%s:%s' %(proxy_ip[0],proxy_ip[1])})
        opener = urllib.request.build_opener(cookies,proxy)
    else:
        opener = urllib.request.build_opener(cookies)
    header = []
    for key, value in head.items():
        elem = (key, value)
        header.append(elem)
    opener.addheaders = header
    return opener

#urllib.request.urlretrieve(url,'/home/gswewf/sougouxibao/下载.scel')
'''
path=sys.path[0]

logging.basicConfig(level=logging.DEBUG,
                format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                datefmt='%a, %d %b %Y %H:%M:%S',
                filename=os.path.join(path,'{}日志.log'.format(time.strftime("%Y%m%d%H%M"))),
                filemode='w+')

def download_link(url,link,ip_list):
    """下载链接文章"""
    file=os.path.join(path,'数据',link[link.find('name=')+5:]+'.scel')
    #若文件所在目录不存在，先新建目录
    if not os.path.isdir(os.path.dirname(file)):
        os.mkdir(os.path.dirname(file))
    if os.path.exists(file):
        #若文件存在则不爬取
        logging.warning("{}已爬取。".format(link))
        return
    #socket.setdefaulttimeout(50)   # 设置超时时间

    opener = urllib.request.build_opener()
    urllib.request.install_opener(opener)

    '''添加默认的 headers.'''
    opener.addheaders = [('User-Agent', 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:22.0) Gecko/20100101 Firefox/22.0'),
                        ('Connection', 'keep-alive'),
                        ('Cache-Control', 'no-cache'),
                        ('Accept-Language', 'zh-CN,zh;q=0.8'),
                        ('Accept-Encoding', 'gzip, deflate,sdch'),
                        ('Accept', 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'),
                        ("Referer",url)]

    '''为 opener 添加 cookiejar handler。'''
    cj = http.cookiejar.CookieJar()
    opener.add_handler(urllib.request.HTTPCookieProcessor(cj))

    for i in range(5):
        try:
            proxy_ip=random.choice(ip_list)
            if proxy_ip:
                # proxy_ip=['176.31.239.33', '8118']
                proxy = urllib.request.ProxyHandler({'http': r'http://%s:%s' %(proxy_ip[0],proxy_ip[1])})
                opener.add_handler(proxy)
            pos=link.find('&name=')+6
            urllib.request.urlretrieve(link[:pos]+urllib.parse.quote(link[pos:]),file)
            logging.warning("完成{}下载".format(link))
            return
        except Exception as e:
            logging.warning("重试下载{}，{}，{}".format(i,link,e))
    logging.warning("{}下载失败".format(link))

def get_download_link(url,ip_list):
    """获取页面中的下载链接"""
    head = {
    "Accept":"text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Encoding":"gzip, deflate, sdch",
    "Accept-Language":"zh-CN,zh;q=0.8",
    #"Connection":"keep-alive",
    "Host":"pinyin.sogou.com",
    "Upgrade-Insecure-Requests":1,
    "User-Agent":"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/49.0.2623.75 Safari/537.36"
    }
    cj = http.cookiejar.CookieJar()
    cookies = urllib.request.HTTPCookieProcessor(cj)
    for i in range(15):
        try:
            proxy_ip=random.choice(ip_list)
            if proxy_ip:
                # proxy_ip=['176.31.239.33', '8118']
                proxy = urllib.request.ProxyHandler({'http': r'http://%s:%s' %(proxy_ip[0],proxy_ip[1])})
                opener = urllib.request.build_opener(cookies,proxy)
            else:
                opener = urllib.request.build_opener(cookies)
            header = []
            for key, value in head.items():
                elem = (key, value)
                header.append(elem)
            opener.addheaders = header

            uop = opener.open(url, timeout = 20)
            html=uop.read()

            links=[]
            datas=''
            try:
                datas=html.decode('utf-8')
            except UnicodeDecodeError:
                #获取网站的编码
                encode=chardet.detect(html)
                #print('网站的编码是：',encode['encoding'])
                if not encode['encoding']:
                    import  zlib
                    data=zlib.decompress(html, 16+zlib.MAX_WBITS) #解决返回网页被压缩了的问题
                    datas=data.decode('utf-8')
            soup = BeautifulSoup(datas,'lxml')
            for div in soup.findAll('div',class_='dict_dl_btn'):
                links.append(div.find('a')['href'])
            return links
        except Exception as e:
            logging.warning("重试爬取链接{}，{}，{}".format(i,url,e))

def get_proxy_ip():
    """获取代理ip"""
    #从json文件加载数据
    file_name=os.path.join(path,'测试通过的代理ip',"{}测试通过的代理服务器ip.json".format(time.strftime("%Y%m%d")))
    #ip_list=[]
    try:
        with open(file_name,'r',encoding='utf-8')as f:
            ip_list=json.loads(f.read())
    except FileNotFoundError:
        import 多线程测试代理IP
        logging.warning('重新测试代理IP。')
        #url='http://index.so.com/#index'
        #keyword='京公网安备110000000006号'
        多线程测试代理IP.main()
        time.sleep(10)
        with open(file_name,'r',encoding='utf-8')as f:
            ip_list=json.loads(f.read())

    return ip_list

def start_pool(url,ip_list):
    links=get_download_link(url,ip_list)
    logging.warning("开始{}".format(url))
    time.sleep(2)
    if not links:
        logging.warning("爬取链接为空{}".format(url))
        return
    with open(os.path.join(path,'通过细胞词库首页网址获取的链接.txt'),'a+',encoding='utf8')as f:
        for link in links:
            f.write(link+'\n')
    for link in links:
        download_link(url,link,ip_list)
        #logging.warning("完成{}".format(link))

def main():
    from multiprocessing.dummy import Pool as ThreadPool
    with open(os.path.join(path,'细胞词库首页网址.json'),encoding='utf8')as f:
        urls=json.load(f)
    ip_list=get_proxy_ip()
    with ThreadPool(4)as pool:
        pool.starmap(start_pool, zip(urls,[ip_list]*len(urls)))


if __name__ == "__main__":
    main()
