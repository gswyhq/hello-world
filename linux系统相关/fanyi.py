#!/usr/bin/python
#-*- coding:UTF-8 -*-

import sys
import re
import json
from bs4 import BeautifulSoup
from urllib.parse import quote
import urllib.request
import http.cookiejar

# 中文-> 英文
zh_en_url = 'https://translate.google.cn/#zh-CN/en/'

# 英文-> 中文
en_zh_url = 'https://translate.google.cn/#en/zh-CN/'

# 命令行翻译使用示例：
# gswewf@gswewf-pc:~$ vim .bash_aliases
# alias fanyi='python3 /home/gswewf/hello-world/linux系统相关/fanyi.py'
# gswewf@gswewf-pc:~$ fanyi python
# gswewf@gswewf-pc:~$ fanyi 你好 中国

def get_text(url='https://translate.google.cn/#en/zh-CN/get%0Aset'):
    from selenium import webdriver
    from selenium.webdriver.common.desired_capabilities import DesiredCapabilities

    dcap = dict(DesiredCapabilities.PHANTOMJS)  # 设置userAgent
    dcap["phantomjs.page.settings.userAgent"] = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.9; rv:25.0) Gecko/20100101 Firefox/25.0 ")
    # 不加载图片
    dcap["phantomjs.page.settings.loadImages"] = False

    service_args = []
    service_args.append('--load-images=no')  ##关闭图片加载
    service_args.append('--disk-cache=yes')  ##开启缓存
    service_args.append('--ignore-ssl-errors=true')  ##忽略https错误

    obj = webdriver.PhantomJS(executable_path='/usr/bin/phantomjs', desired_capabilities=dcap,
                              service_args=service_args)  # 加载网址

    # 隐式等待5秒，可以自己调节
    obj.implicitly_wait(5)
    # 设置10秒页面超时返回，类似于requests.get()的timeout选项，driver.get()没有timeout选项
    # 以前遇到过driver.get(url)一直不返回，但也不报错的问题，这时程序会卡住，设置超时选项能解决这个问题。
    obj.set_page_load_timeout(10)
    # 设置10秒脚本超时时间
    obj.set_script_timeout(10)

    obj.maximize_window()  # 设置全屏

    obj.get(url)  # 打开网址
    # obj.save_screenshot("1.png")  # 截图保存

    # data = obj.page_source
    # print(data)
    # print(obj.title)
    # print(obj.find_element_by_xpath('//*[@id="result_box"]'))
    # print(obj.find_element_by_id("result_box").text)
    text = obj.find_element_by_xpath('//*[@id="result_box"]').text
    obj.quit()  # 关闭浏览器。当出现异常时记得在任务浏览器中关闭PhantomJS，因为会有多个PhantomJS在运行状态，影响电脑性能
    print(text)
    return text

def open_chrome(url):
    from selenium import webdriver
    options = webdriver.ChromeOptions()
    options.add_experimental_option("excludeSwitches", ["ignore-certificate-errors"])
    browser = webdriver.Chrome(chrome_options=options)
    # browser = webdriver.Firefox()

    browser.get(url)
    # browser.find_element_by_id("kw").send_keys("selenium")
    # browser.find_element_by_id("su").click()
    # time.sleep(3)  # 休眠3秒
    browser.quit()

def main():
    if len(sys.argv) > 1:
        argv_list = sys.argv[1:]
    else:
        argv_list = []

    if all(re.search('^([a-zA-Z])+$', t) for t in argv_list):
        url = en_zh_url
    else:
        url = zh_en_url

    url = url + quote(' '.join(argv_list))

    # 直接通过获取翻译结果， 但是速度太慢
    # get_text(url)
    # 打开浏览器
    open_chrome(url)


if __name__ == '__main__':
    main()