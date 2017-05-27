#/usr/lib/python3.5
# -*- coding: utf-8 -*-
'''伪装浏览器
使用chrome浏览器自带的开发者工具查看http头的方法
1.在网页任意地方右击选择审查元素或者按下 shift+ctrl+c, 打开chrome自带的调试工具;
2.选择network标签, 刷新网页(在打开调试工具的情况下刷新);
3.刷新后在左边找到该网页url,点击 后右边选择headers,就可以看到当前网页的http头了;
请求Header(HTTP request header )
Host 请求的域名
User-Agent 浏览器端浏览器型号和版本
Accept 可接受的内容类型
Accept-Language 语言
Accept-Encoding 可接受的压缩类型 gzip,deflate
Accept-Charset 可接受的内容编码 UTF-8,*
服务器端的响应Header(response header)
Date 服务器端时间
Server 服务器端的服务器软件 Apache/2.2.6
Etag 文件标识符
Content-Encoding传送启用了GZIP压缩 gzip
Content-Length 内容长度
Content-Type 内容类型

最近和同事一起看Web的Cache问题，又进一步理解了 HTTP 中的 304 又有了一些了解。 304 的标准解释是：

Not Modified 客户端有缓冲的文档并发出了一个条件性的请求（一般是提供If-Modified-Since头表示客户只想比指定日期更新的文档）。服务器告诉客户，原来缓冲的文档还可以继续使用。
如果客户端在请求一个文件的时候，发现自己缓存的文件有 Last Modified ，那么在请求中会包含 If Modified Since ，这个时间就是缓存文件的 Last Modified 。因此，如果请求中包含 If Modified Since，就说明已经有缓存在客户端。只要判断这个时间和当前请求的文件的修改时间就可以确定是返回 304 还是 200 。对于静态文件，例如：CSS、图片，服务器会自动完成 Last Modified 和 If Modified Since 的比较，完成缓存或者更新。但是对于动态页面，就是动态产生的页面，往往没有包含 Last Modified 信息，这样浏览器、网关等都不会做缓存，也就是在每次请求的时候都完成一个 200 的请求。

因此，对于动态页面做缓存加速，首先要在 Response 的 HTTP Header 中增加 Last Modified 定义，其次根据 Request 中的 If Modified Since 和被请求内容的更新时间来返回 200 或者 304 。虽然在返回 304 的时候已经做了一次数据库查询，但是可以避免接下来更多的数据库查询，并且没有返回页面内容而只是一个 HTTP Header，从而大大的降低带宽的消耗，对于用户的感觉也是提高。

当这些缓存有效的时候，通过 HttpWatch 查看一个请求会得到这样的结果：
第一次访问 200
鼠标点击二次访问 (Cache)
按F5刷新 304
按Ctrl+F5强制刷新 200

如果是这样的就说明缓存真正有效了。以上就是我对 HTTP 304 的一个理解。
'''
def fangfa1():
    #在 GET 的时候添加 header 有很多方法, 下面介绍两种方法.

    #第一种方法比较简便直接, 但是不好扩展功能, 代码如下:

    import urllib.request

    url = 'http://www.youdaili.net/Daili/http/4261.html'
    req = urllib.request.Request(url, headers = {
        'age':0,
 'cache-control':'max-age=14400',
 'Connection':'keep-alive',
 'Content-Encoding':'gzip',
 'Content-Type':'text/html',
 'Server':'nginx',
 #'Transfer-Encoding':'chunked',
 'VAR-Cache':'HIT',
 'Vary':'Accept-Encoding',
 'X-Powered-By-360WZB':'wangzhan.360.cn',
 'Accept':'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
 'Accept-Encoding':'gzip, deflate, sdch',
 'Accept-Language':'zh-CN,zh;q=0.8',
 'Cache-Control':'no-cache',
 'Connection':'keep-alive',
 'Cookie':'BAIDU_SSP_lcr=https://www.baidu.com/link?url=INJDtsS5F0NMMN0cOEGMxGhJs08TzgLxypWI0NXFxamOwstaIGkSxMWSyN8skiPP&wd=&eqid=b241e8da0026b1f40000000656eb655c; __utma=254986240.983134157.1458267709.1458267709.1458267709.1; __utmc=254986240; __utmz=254986240.1458267709.1.1.utmcsr=(direct)|utmccn=(direct)|utmcmd=(none); Hm_lvt_f8bdd88d72441a9ad0f8c82db3113a84=1458267543; Hm_lpvt_f8bdd88d72441a9ad0f8c82db3113a84=1458274674',
 'Host':'www.youdaili.net',
 'Pragma':'no-cache',
 'Upgrade-Insecure-Requests':1,
 'User-Agent':'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/49.0.2623.75 Safari/537.36'})
    oper = urllib.request.urlopen(req)
    data = oper.read()
    import  zlib
    html=zlib.decompress(data, 16+zlib.MAX_WBITS);
    print(html)
    print(html.decode('utf-8'))

def fangfa2():
    #第二种方法使用了 build_opener 这个方法, 用来自定义 opener, 这种方法的好处是可以方便的拓展功能,
    # 例如下面的代码就拓展了自动处理�0�2Cookies 的功能.

    import urllib.request
    import http.cookiejar

    # head: dict of header
    def makeMyOpener(head = {
        'Connection': 'Keep-Alive',
        'Accept': 'text/html, application/xhtml+xml, */*',
        'Accept-Language': 'zh-CN,zh;q=0.8,en-US,en;q=0.8,zh-Hans-CN;q=0.5,zh-Hans;q=0.3',
        'User-Agent': 'Mozilla/5.0 (Windows NT 6.3; WOW64; Trident/7.0; rv:11.0) like Gecko',
        'cache-control':'no-cache',
        'age':0,
        'VAR-Cache':'HIT',
        'User-Agent':'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/49.0.2623.75 Safari/537.36'
        }):
        cj = http.cookiejar.CookieJar()
        opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
        header = []
        for key, value in head.items():
            elem = (key, value)
            header.append(elem)
        opener.addheaders = header
        return opener

    oper = makeMyOpener()
    uop = oper.open('http://www.youdaili.net/Daili/http/4261.html', timeout = 1000)
    data = uop.read()
    print(data.decode())

def main():
    fangfa1()


if __name__ == "__main__":
    main()