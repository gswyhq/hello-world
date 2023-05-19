#!/usr/bin/python
# -*- coding:UTF-8 -*-


import os

USERNAME = os.getenv("USERNAME")
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

# 新版的 selenium已经放弃PhantomJS改用Chorme headless
# 如果还想继续用PhantomJS的话只能使用旧版的selenium，卸载之后重新pip install selenium==2.48.0安装成功。
# 但其实更好的选择，我们可以使用firefox或chrome的headlesss模式,无需重装selenium
# 也能完美实现无界面爬虫

def test1(url='https://www.baidu.com/'):
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--disable-gpu')#上面三行代码就是为了将Chrome不弹出界面，实现无界面爬取


    #设置user-agent
    chrome_options.add_argument('user-agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10.9; rv:25.0) Gecko/20100101 Firefox/25.0 "')

    driver = webdriver.Chrome(executable_path=rf"D:\Users\{USERNAME}\chromedriver_win32/chromedriver.exe", chrome_options=chrome_options)

    # 隐式等待5秒，可以自己调节
    driver.implicitly_wait(5)
    # 设置10秒页面超时返回，类似于requests.get()的timeout选项，driver.get()没有timeout选项
    # 以前遇到过driver.get(url)一直不返回，但也不报错的问题，这时程序会卡住，设置超时选项能解决这个问题。
    driver.set_page_load_timeout(10)
    # 设置10秒脚本超时时间
    driver.set_script_timeout(10)

    driver.maximize_window()  # 设置全屏

    driver.get(url)

    data = driver.page_source
    print(data)
    print(data.title)
    print('京ICP证030173号' in data)
    driver.quit()

def main():
    test1()


if __name__ == '__main__':
    main()



