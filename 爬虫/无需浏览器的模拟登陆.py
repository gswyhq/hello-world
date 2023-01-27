#!/usr/bin/python
# -*- coding:UTF-8 -*-


def test1(url='https://www.baidu.com/'):
    from selenium import webdriver
    # PhantomJS 是一个基于 WebKit（WebKit是一个开源的浏览器引擎，Chrome，Safari就是用的这个浏览器引擎） 的服务器端 JavaScript API，
    # 主要应用场景是：无需浏览器的 Web 测试，页面访问自动化，屏幕捕获，网络监控
    # sudo apt-get install phantomjs
    # gswewf@gswewf-pc:~$ whereis phantomjs
    # phantomjs: /usr/bin/phantomjs /usr/lib/phantomjs /usr/share/man/man1/phantomjs.1.gz
    driver = webdriver.PhantomJS(executable_path='/usr/bin/phantomjs')
    # obj = webdriver.PhantomJS(executable_path='C:\Python27\Scripts\phantomjs.exe') #windows

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


def test2(url='https://translate.google.cn/#en/zh-CN/get%0Aset'):
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

    data = obj.page_source
    print(data)
    print(obj.title)
    print(obj.find_element_by_xpath('//*[@id="result_box"]'))
    print(obj.find_element_by_id("result_box").text)
    obj.quit()  # 关闭浏览器。当出现异常时记得在任务浏览器中关闭PhantomJS，因为会有多个PhantomJS在运行状态，影响电脑性能


def test3():
    from selenium import webdriver

    driver = webdriver.PhantomJS()
    driver.get('http://stackoverflow.com/')

    cookies = driver.get_cookies()

    driver.delete_all_cookies()

    for cookie in cookies:
        driver.add_cookie({k: cookie[k] for k in ('name', 'value', 'domain', 'path', 'expiry')})


def getResponseHeaders(driver):
    """解析出headers,如：
    {'Age': '0',
     'Via': 'http/1.1 ORI-CLOUD-JN2-MIX-116 (jcs [cMsSf ]), http/1.1 ZJ-CT-1-MIX-11 (jcs [cMsSf ])',
     'Date': 'Fri, 08 Jan 2021 02:37:09 GMT',
     'Pragma': 'no-cache',
     'Server': 'nginx',
     'Expires': 'Fri, 08 Jan 2021 02:37:09 GMT',
     'X-Trace': '200;200-1610073429318-0-0-0-629-629;200-1610073429317-0-0-0-644-644',
     'Connection': 'Keep-Alive',
     'Content-Type': 'text/html;charset=UTF-8',
     'Cache-Control': 'max-age=0',
     'Content-Language': 'zh-CN',
     'Transfer-Encoding': 'chunked',
     'Strict-Transport-Security': 'max-age=360'}
    """
    har = json.loads(driver.get_log('har')[0]['message'])
    return {header["name"]: header["value"] for header in har['log']['entries'][0]['response']["headers"]}

def getResponseStatus(driver):
    """
    返回状态码和状态
    :param driver:
    :return: (200, 'OK')
    """
    har = json.loads(driver.get_log('har')[0]['message'])
    return (har['log']['entries'][0]['response']["status"], str(har['log']['entries'][0]['response']["statusText"]))

def main():
    test1()


if __name__ == '__main__':
    main()


# 有时，phantomJS获得的页面源码的确存在某元素，但通过find_element_by_xpath()等定位函数却无法获得该元素对象，
# 总是提示“元素不存在”的错误。遇到这种情况，除了检查元素节点路径是否正确外，还应该分析页面源码，
# 检查元素是否被包裹在一个特定的frame中，如果是后者，那么在使用查找函数前，需要额外的处理。
#
# 比如网页源码中有如下代码:
#
# <iframe id="topmenuFrame" width="100%" scrolling="no" height="100%" src="topmenu.aspx?>
# <div id="haha">text</div>
# </iframe>
# 假如你想要获取id="haha"的div标签，直接通过driver.find_element_by_id('haha')就会提示“元素不存在“的错误。
#
# 这时需要使用driver.switch_to_frame(driver.find_element_by_id``("topmenuFrame"))，即先进入id为topmenuFrame的frame，
# 然后再执行driver.find_element_by_id("haha")，就能正确获得该元素了。
#
# 需要注意的是，切换到这个frame之后，只能访问当前frame的内容，如果想要回到默认的内容范围，相当于默认的frame，
# 还需要使用driver.switch_to_default_content()。
# 页面中有多个frame时，要注意frame之间的切换。
