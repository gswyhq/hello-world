#!/usr/bin/python3
# coding: utf-8

import json
from urllib.request import urlopen
from urllib.request import Request
import random
import re
import os
import time
from urllib import parse
# from multiprocessing import Pool
from bs4 import BeautifulSoup
from multiprocessing.dummy import Pool as ThreadPool
from zhconv import convert

with open('/home/gswyhq/下载/爬取的网址信息.json', 'r')as f:
    ts3 = json.load(f)

PATH = '/home/gswyhq/下载/html'

def getContent(url, headers):
    """
    此函数用于抓取返回403禁止访问的网页
    """
    random_header = random.choice(headers)
    """
    对于Request中的第二个参数headers，它是字典型参数，所以在传入时
    也可以直接将个字典传入，字典中就是下面元组的键值对应
    """
    # url解码
    urldata = parse.unquote(url)

    # url结果
    result = parse.urlparse(urldata)
    netloc = result.netloc


    req = Request(url)
    req.add_header("User-Agent", random_header)
    req.add_header("GET", url)
    req.add_header("Host", netloc)
    req.add_header("Referer", "http://{}/".format(netloc))
    html=urlopen(req, timeout=5).read()
    print(time.time(), type(html), html[:300])
    try:
        if b'charset=utf' in html.lower():
            errors='ignore'
        else:
            errors='strict'
        content = html.decode("utf-8", errors=errors)
    except UnicodeDecodeError:
        if b'charset=gbk' in html.lower():
            errors='ignore'
        else:
            errors='strict'
        content = html.decode("gbk", errors=errors)
    return content

def save_html_to_file(title, url):
    my_headers = [
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/49.0.2623.75 Safari/537.36"]
    save_file = os.path.join(PATH, "{}.html".format(title))
    if os.path.isfile(save_file):
        return ''
    html = getContent(url, my_headers)
    with open(save_file, 'w')as f:
        print(html, file=f)

# url = "http://www.baidu.com"
# # 这里面的my_headers中的内容由于是个人主机的信息，所以我就用句号省略了一些，在使用时可以将自己主机的
# my_headers = ["Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/49.0.2623.75 Safari/537.36"]
# print(getContent(url, my_headers))

def html_to_txt(title):
    """将下载的html文件转换为txt"""
    html_file = os.path.join(PATH, "{}.html".format(title))
    save_file = os.path.join(PATH, "txt/{}.txt".format(title))
    with open(html_file)as f:
        content = f.read()
    soup = BeautifulSoup(content, "lxml")
    with open(save_file, 'w', encoding='utf-8')as f:
        for text in soup.find_all('div', id='content'):
            for t in text.strings:
                t = convert(t.strip(), 'zh-cn') # 繁体转简体
                print(t, file=f)

def save_html_to_file_job1(z):
    # pool.map不能处理多参数；只能通过对有多个参数的方法进行封装
    return save_html_to_file(z[0], z[1])

def main():
    # [save_html_to_file(title, url) for url, title in ts3]
    # with ThreadPool() as pool:
    #     pool.map(save_html_to_file_job1, [(title, url) for url, title in ts3])
    [html_to_txt(title) for url, title in ts3]

if __name__ == '__main__':
    main()