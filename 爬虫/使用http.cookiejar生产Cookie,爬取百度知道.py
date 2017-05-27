#conding:utf-8
# usr/bin/python3.5
#使用http.cookiejar生产Cookie,爬取百度知道

#http://cuiqingcai.com/968.html
#http://blog.csdn.net/keenweiwei/article/details/9041307
#http://blog.csdn.net/samxx8/article/details/21535901
#http://www.zhaoyoucai.com/86.html

#
import os
import urllib.request
import http.cookiejar
from bs4 import BeautifulSoup
import chardet

def get_cookie(url='http://zhidao.baidu.com/question/75367260.html',write=0):
    """将cookie保存到变量中，然后打印出了cookie中的值"""
    #声明一个CookieJar对象实例来保存cookie
    cookie=http.cookiejar.CookieJar()
    #利用urllib2库的HTTPCookieProcessor对象来创建cookie处理器
    handler=urllib.request.HTTPCookieProcessor(cookie)
    #通过handler来构建opener
    opener=urllib.request.build_opener(handler)

    #此处的open方法同urllib2的urlopen方法，也可以传入request
    response=opener.open(url)

    for item in cookie:
        print('{}={}'.format(item.name,item.value))

def save_cookie(url='http://zhidao.baidu.com/question/75367260.html',
            filename='/home/gswyhq/gow69/cookie.txt'):
    """保存Cookie到文件"""
    #设置保存cookie的文件，同级目录下的cookie.txt
    
    #声明一个MozillaCookieJar对象实例来保存cookie，之后写入文件
    cookie=http.cookiejar.MozillaCookieJar(filename)
    #HTTPCookieProcessor对象来创建cookie处理器
    handler=urllib.request.HTTPCookieProcessor(cookie)

    opener=urllib.request.build_opener(handler)

    response=opener.open(url)

    #保存cookie到文件
    cookie.save(ignore_discard=True, ignore_expires=True)
    # ignore_discard的意思是即使cookies将被丢弃也将它保存下来，
    # ignore_expires的意思是如果在该文件中cookies已经存在，则覆盖原文件写入

def  get_cookie_openurl(url='http://zhidao.baidu.com/question/75367260.html',
            filename='/home/gswyhq/gow69/cookie.txt'):
    """从文件中获取Cookie并访问"""
 
    #创建MozillaCookieJar实例对象
    cookie = http.cookiejar.MozillaCookieJar()
    #从文件中读取cookie内容到变量
    cookie.load(filename, ignore_discard=True, ignore_expires=True)
    #创建请求的request
    req = urllib.request.Request(url)
    #利用urllib2的build_opener方法创建一个opener
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cookie))
    response = opener.open(req)
    source_code=response.read()
    
    #获取网站的编码
    encode=chardet.detect(source_code)
    #print('网站的编码是：',encode['encoding'])
    try:
        return source_code.decode(encode['encoding'])
    except UnicodeDecodeError:
        return source_code.decode('gbk')
    
def parse_html(html):
    """解析html"""

    # Beautiful Soup 是用Python写的一个HTML/XML的解析器，
    #它可以很好的处理不规范标记并生成剖析树(parse tree)。
    soup = BeautifulSoup(html,'lxml')
    
    #定位到所有需解析的内容
    qt_content = soup.find('article', {'class': 'grid qb-content','id':'qb-content'})#通过属性进行查找
    #解析问题部分
    wgt_ask=qt_content.find('div', id='wgt-ask')
    qcontent=wgt_ask.find('h1').get_text()#问题标题
    for q in wgt_ask.findAll('pre'):
        #问题的详述
        qcontent+=q.get_text()
    
    answer_content=''    
    #网友回答部分,采纳的答案
    answer=qt_content.find('div', class_='bd answer')
    if answer:
        answer=answer.find('pre')
        answer_content+=answer.get_text()
    #专业回答
    answer=qt_content.find('div', class_='quality-content-detail content')
    if answer:
        answer_content+=answer.get_text()
    #其他的答案
    other_answer=qt_content.find('div', class_='bd-wrap')
    if other_answer:
        other_answer=other_answer.findAll('div', class_='answer-text line')
        for a in other_answer:
            answer_content+=a.get_text()
    
    return qcontent, answer_content    
    
if __name__=="__main__":
    url='http://zhidao.baidu.com/question/744456942700491092.html'
    save_cookie(url)
    html=get_cookie_openurl(url)
    file =os.path.join('/home/gswyhq/gow69/',os.path.split(url)[-1])
    with open (file,'w',encoding='utf-8')as f:
        print(html,  file=f)
    ask, answer=parse_html(html)
    print(ask)
    print('*'*8)
    print(answer)




