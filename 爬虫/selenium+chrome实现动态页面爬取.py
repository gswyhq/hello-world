
# selenium+chrome实现动态页面爬取

from selenium import webdriver
import time
import re
import json, os
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities


# driver.get()这个操作，改成不阻塞的就行了，这样打开网页就操作完成了，不需要等他加载
#get直接返回，不再等待界面加载完成
# 配置一个参数，就是页面加载策略，系统默认是等待，就是等他加载完，直接设置成none，就是不等待，这样就是get操作完后直接就是结束了
desired_capabilities = DesiredCapabilities.CHROME
desired_capabilities["pageLoadStrategy"] = "none"

def url_to_html_file(url):
    return '_'.join(re.findall('[a-zA-Z0-9]+', url)) + '.html'

# Chrome与chromedriver.exe的版本对应
'''
第一步： 查看Chrome浏览版本
打开chrome浏览器，输入chrome://version/
如：
Google Chrome	71.0.3578.98 (正式版本) （32 位） (cohort: Stable)
主要看“71.0.3578.98”第一个数字（这里是71），根据对应数字找到对应的chromedriver版本

第二步： 进入Chromedrvier下载地址
下载地址1：http://npm.taobao.org/mirrors/chromedriver/
下载地址2：http://chromedriver.storage.googleapis.com/index.html
查看对应的notes.txt文件，或者根据ChromeDriver选择最近日期的对应版本（如ChromeDriver v2.45 -> Chrome v70-72）即可；
'''
# 启动浏览器
url='http://kaoshi.edu.sina.com.cn/college/scorelist?tab=major&majorid=&wl=&local=5&provid=&batch=&syear=2017&page=1'
# url = 'https://www.baidu.com'
driver=webdriver.Chrome(r'D:\Users\user123\chromedriver_win32\chromedriver.exe')
# 以前遇到过driver.get(url)一直不返回，但也不报错的问题，这时程序会卡住，设置超时选项能解决这个问题。
# driver.set_page_load_timeout(30)
# 设置10秒页面超时返回，类似于requests.get()的timeout选项，driver.get()没有timeout选项
# driver.set_script_timeout(10)

# driver.set_page_load_timeout(5)这玩意一设置，没问题，5秒后网页确实停止了，但是driver也死了，不管运行什么都是timeout，还说try一下，driver是死透了，只能重新来

# 注意：使用set_page_load_timeout时候，当页面未加载出任何东西的时候（往往是html源码未加载），因为超时而停止，会导致driver失效，
# 后面的driver都不能操作，所以超时设置应该至少保证页面内容加载出来一部分，设置超时不宜过短，如下图在页面此种状态下停止加载后driver失效。

driver.get(url)
# 注：这里的url,不一定是http开头的网址，也可以是mht、html文件地址,这时获取的内容，就是渲染后的HTML文件内容；

time.sleep(15)
# driver.implicitly_wait(10)
driver.maximize_window() # 最大化窗口
# driver.implicitly_wait(10)
# driver.find_element_by_link_text('热点').click()  # 点击热点
# driver.implicitly_wait(10)
# driver.implicitly_wait(10)


data = driver.page_source

save_html2 = r'D:\Users\user123\data\各高校及专业排名\新浪教育\html'
save_file = os.path.join(save_html2, url_to_html_file(url))

with open(save_file, 'w', encoding='utf-8')as f:
    f.write(data)

driver.close()
time.sleep(1)


def get_html_chrome(url):
    from selenium import webdriver
    from selenium.webdriver.common.desired_capabilities import DesiredCapabilities

    # get直接返回，不再等待界面加载完成
    desired_capabilities = DesiredCapabilities.CHROME
    desired_capabilities["pageLoadStrategy"] = "none"

    # 引入 ActionChains 类
    from selenium.webdriver.common.action_chains import ActionChains

    # dcap = dict(DesiredCapabilities.PHANTOMJS)  # 设置userAgent
    # dcap["phantomjs.page.settings.userAgent"] = (
    #     "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.9; rv:25.0) Gecko/20100101 Firefox/25.0 ")
    # # 不加载图片
    # dcap["phantomjs.page.settings.loadImages"] = False

    service_args = []
    service_args.append('--load-images=no')  ##关闭图片加载
    service_args.append('--disk-cache=yes')  ##开启缓存
    service_args.append('--ignore-ssl-errors=true')  ##忽略https错误

    # chromedriver 下载地址：https://cdn.npmmirror.com/binaries/chromedriver/112.0.5615.49/chromedriver_linux64.zip
    # 注意跟本机Chrome浏览器版本匹配
    obj = webdriver.Chrome(executable_path='/home/gswyhq/Downloads/chromedriver_112_0_5615_49/chromedriver',
                           desired_capabilities=desired_capabilities,
                           service_args=service_args
                           )  # 加载网址

    # 隐式等待5秒，可以自己调节
    # obj.implicitly_wait(15)
    # 设置10秒页面超时返回，类似于requests.get()的timeout选项，driver.get()没有timeout选项
    # 以前遇到过driver.get(url)一直不返回，但也不报错的问题，这时程序会卡住，设置超时选项能解决这个问题。
    # obj.set_page_load_timeout(30)
    # 设置10秒脚本超时时间
    # obj.set_script_timeout(30)

    obj.maximize_window()  # 设置全屏

    obj.get(url)  # 打开网址
    time.sleep(10)
    # obj.save_screenshot("1.png")  # 截图保存

    # 目前所有页面的句柄
    # print('页面句柄', obj.window_handles)

    # 页面句柄切换
    # driver.switch_to.window(handles[1])
    obj.switch_to.frame("play")  # #切换到name为play的iframe中
    time.sleep(2)
    # 定位到要悬停的元素
    obj.find_element_by_xpath('//*[@id="thisbody"]')
    above = obj.find_element_by_link_text("刷新")

    data = obj.page_source
    for _ in range(3):
        if '.mp3?key' in data or '.m4a?key' in data:
            break
        else:
            time.sleep(10)
            data = obj.page_source
    obj.quit()  # 关闭浏览器。当出现异常时记得在任务浏览器中关闭PhantomJS，因为会有多个PhantomJS在运行状态，影响电脑性能
    if '.mp3?key' in data or '.m4a?key' in data:
        return data
    else:
        return ''

def main():
    url = 'https://www.70ts.com/tingshu/11276/55234.html'
    get_html_chrome(url)


if __name__ == '__main__':
    main()