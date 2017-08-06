#!/usr/bin/python3
# coding: utf-8

# 安装 pychrome:
# 从github安装：
# $ pip install -U git+https://github.com/fate0/pychrome.git
#
# 或者从pip源安装:
# $ pip install -U pychrome
#
# 或者从源码进行安装：
# $ python setup.py install
#
# 设置chrome
# 简单地：
# $ google-chrome --remote-debugging-port=9222
#
# 或者交互模式 (chrome version >= 59):
# $ google-chrome --headless --disable-gpu --remote-debugging-port=9222
#
# 或者使用docker：
# $ docker pull fate0/headless-chrome
# $ docker run -it --rm --cap-add=SYS_ADMIN -p9222:9222 fate0/headless-chrome



# 实例1：
# create a browser instance
browser = pychrome.Browser(url="http://127.0.0.1:9222")

# list all tabs (default has a blank tab)
tabs = browser.list_tab()

if not tabs:
    tab = browser.new_tab()
else:
    tab = tabs[0]


# register callback if you want
def request_will_be_sent(**kwargs):
    print("loading: %s" % kwargs.get('request').get('url'))

tab.Network.requestWillBeSent = request_will_be_sent

# call method
tab.Network.enable()
# call method with timeout
tab.Page.navigate(url="https://github.com/fate0/pychrome", _timeout=5)

# 6. wait for loading
tab.wait(5)

# 7. stop tab (stop handle events and stop recv message from chrome)
tab.stop()

# 8. close tab
browser.close_tab(tab)


# 示例2：
browser = pychrome.Browser(url="http://127.0.0.1:9222")

tabs = browser.list_tab()
if not tabs:
    tab = browser.new_tab()
else:
    tab = tabs[0]


def request_will_be_sent(**kwargs):
    print("loading: %s" % kwargs.get('request').get('url'))


tab.set_listener("Network.requestWillBeSent", request_will_be_sent)

tab.call_method("Network.enable")
tab.call_method("Page.navigate", url="https://github.com/fate0/pychrome", _timeout=5)

tab.wait(5)
tab.stop()

browser.close_tab(tab)

# 更多事件和方法可以在下面地址找到
# https://chromedevtools.github.io/devtools-protocol/tot/

# Tab 管理：

# $ pychrome new http://www.fatezero.org
# {
#     "description": "",
#     "url": "http://www.fatezero.org/",
#     "webSocketDebuggerUrl": "ws://127.0.0.1:9222/devtools/page/557d8315-e909-466c-bf20-f5a6133ebd89",
#     "id": "557d8315-e909-466c-bf20-f5a6133ebd89",
#     "type": "page",
#     "devtoolsFrontendUrl": "/devtools/inspector.html?ws=127.0.0.1:9222/devtools/page/557d8315-e909-466c-bf20-f5a6133ebd89",
#     "title": ""
# }
#
# $ pychrome close 557d8315-e909-466c-bf20-f5a6133ebd89
# Target is closing
#
#
# 更多使用实例（https://github.com/fate0/pychrome/tree/master/examples）

def main():
    pass


if __name__ == '__main__':
    main()