#!/usr/bin/python3
# coding: utf-8


import urllib, urllib.request
import sys
import time

class DefaultErrorHandler(urllib.request.HTTPDefaultErrorHandler):
    """用来保证请求中记录Http状态
"""
    def http_error_default(self, req, fp, code, msg, headers):
        result = urllib.request.HTTPError(
            req.get_full_url(), code, msg, headers, fp)
        result.status = code
        return result

class Sample():
    """想下载的某个资源"""
    url = None
    contentLength = 0
    etag = None
    lastModified = None
    data = None
    path = None

    def __init__(self, url, contentLength=0, etag=None, lastModified=None):
        self.url = url
        self.contentLength = contentLength
        self.etag = etag
        self.lastModified = lastModified
        self.status = 200

    def __repr__(self):
        # print("self.etag: `{}`".format(self.etag))
        return repr("Http Status={}; Length={}; Last Modified Time={}; eTag={}".format(self.status, self.contentLength, self.lastModified, self.etag))

    def downloadSample(self):
        request = urllib.request.Request(self.url)
        request.add_header('User-Agent', 'MyHTTPWebServicesApp/1.0')
        # request.add_header('User-Agent', "Mozilla/5.0 (Macintosh; U; PPC Mac OS X; en)")
        if self.lastModified:
            request.add_header('If-Modified-Since', self.lastModified)
        if self.etag:
            print("设置发送etag: `{}`".format(self.etag))
            request.add_header('If-None-Match', self.etag)
        conn = urllib.request.build_opener(DefaultErrorHandler()).open(request)

        if hasattr(conn, 'headers'):
            # save ETag, if the server sent one
            self.etag = conn.headers.get('ETag')
            # save Last-Modified header, if the server sent one
            self.lastModified = conn.headers.get('Last-Modified')

            self.contentLength = conn.headers.get("content-length")

        if hasattr(conn, 'status'):
            self.status = conn.status
            print("status=%d" % self.status)

        self.data = conn.read()

        if self.status == 304:
            print("内容是一样的，所以没有任何回报！")

        if not self.contentLength:
            self.contentLength = len(self.data)

        conn.close()

def main():
    # url = 'http://www.sina.com.cn'
    # url = 'https://www.baidu.com/'
    url = 'http://192.168.3.101/web1/chat_bot_demo/raw/xincheng_zhongjixian/input/xincheng/kg_xincheng_zhongjixian.txt'
    # url = 'https://raw.githubusercontent.com/bells/elasticsearch-analysis-dynamic-synonym/master/pom.xml'
    sample = Sample(url)
    sample.downloadSample()
    print(sample)
    sample.downloadSample()
    print(sample)


if __name__ == '__main__':
    main()

# http://www.sina.com.cn
# status = 200
# 'Http Status=200; Length=603551; Last Modified Time=Fri, 15 Dec 2017 09:09:14 GMT; eTag=None'
# status = 304
# 内容是一样的，所以没有任何回报！
# 'Http Status=304; Length=0; Last Modified Time=None; eTag=None'

# https://www.baidu.com/
# status=200
# 'Http Status=200; Length=227; Last Modified Time=Thu, 07 Dec 2017 06:53:00 GMT; eTag=None'
# status=200
# 'Http Status=200; Length=227; Last Modified Time=Thu, 07 Dec 2017 06:53:00 GMT; eTag=None'

# Cache-Control:max-age=60, private
# Connection:keep-alive
# Content-Disposition:inline
# Content-Encoding:gzip
# Content-Type:text/plain; charset=utf-8
# Date:Fri, 15 Dec 2017 09:15:52 GMT
# Etag:W/"df900e8075166522deddbbc44a7173b2"
# Server:nginx
# Transfer-Encoding:chunked
# X-Content-Type-Options:nosniff
# X-Frame-Options:DENY
# X-Request-Id:15378dc3-fd21-4c17-becb-f9bd207cd69c
# X-Runtime:0.039198
# X-Ua-Compatible:IE=edge
# X-Xss-Protection:1; mode=block
# Request Headers
# view source
# Accept:text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8
# Accept-Encoding:gzip, deflate, sdch
# Accept-Language:zh-CN,zh;q=0.8
# Cache-Control:max-age=0
# Connection:keep-alive
# Cookie:_gitlab_session=822cdd7f7253b6a81bce949d178caa2f
# Host:192.168.3.101
# If-None-Match:W/"df900e8075166522deddbbc44a7173b2"
# Referer:http://192.168.3.101/web1/chat_bot_demo/blob/xincheng_zhongjixian/input/xincheng/kg_xincheng_zhongjixian.txt
# Upgrade-Insecure-Requests:1
# User-Agent:Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36

# 你已经看到了构造一个职能的 HTTP web 客户的所有片断。 现在让我们看看如何将它们整合到一起。
# 例 11.17. openanything 函数
# 这个函数定义在 openanything.py 中。
# def openAnything(source, etag=None, lastmodified=None, agent=USER_AGENT):
#     # non-HTTP code omitted for brevity
#     if urlparse.urlparse(source)[0] == 'http':  1
#         # open URL with urllib2
#         request = urllib2.Request(source)
#         request.add_header('User-Agent', agent)  2
#         if etag:
#             request.add_header('If-None-Match', etag)  3
#         if lastmodified:
#             request.add_header('If-Modified-Since', lastmodified)  4
#         request.add_header('Accept-encoding', 'gzip')  5
#         opener = urllib2.build_opener(SmartRedirectHandler(), DefaultErrorHandler()) 6        return opener.open(request)  7
# 1	urlparse 是一个解析 URL 的垂手可得的工具模块。 它的主要功能也调用 urlparse, 获得一个 URL 并将其拆分为为一个包含 (scheme, domain, path, params, 查询串参数 和 验证片断) 的 tuple。 当然, 你唯一需要注意的就是 scheme, 确认你处理的是一个 HTTP URL (urllib2 才能处理)。
# 2	通过调用函数使用 User-Agent 向 HTTP 服务器确定你的身份。 如果没有 User-Agent 被指定, 你会使用一个默认的, 就是定义在早期的 openanything.py 模块中的那个。 你从来不会使用到默认的定义在 urllib2 中的那个。
# 3	如果给出了 ETag, 要在 If-None-Match 头信息中发送它。
# 4	如果给出了最近修改日期, 要在 If-Modified-Since 头信息中发送它。
# 5	如果可能要告诉服务器你要获取压缩数据。
# 6	使用 两个 自定义 URL 处理器创建一个 URL 开启器: SmartRedirectHandler 为了处理 301 和 302 重定向, 而 DefaultErrorHandler 为了处理 304, 404 以及其它的错误条件。
# 7	就这样! 打开 URL 并返回一个类似文件的对象给调用者。
# 例 11.18. fetch 函数
# 这个函数定义在 openanything.py 中。
# def fetch(source, etag=None, last_modified=None, agent=USER_AGENT):
#     '''Fetch data and metadata from a URL, file, stream, or string'''
#     result = {}
#     f = openAnything(source, etag, last_modified, agent)  1
#     result['data'] = f.read()  2
#     if hasattr(f, 'headers'):
#         # save ETag, if the server sent one
#                result['etag'] = f.headers.get('ETag')  3
#         # save Last-Modified header, if the server sent one
#         result['lastmodified'] = f.headers.get('Last-Modified')  4
#         if f.headers.get('content-encoding', '') == 'gzip':  5
#                         # data came back gzip-compressed, decompress it
#             result['data'] = gzip.GzipFile(fileobj=StringIO(result['data']])).read()
#     if hasattr(f, 'url'):  6
#         result['url'] = f.url
#         result['status'] = 200
#     if hasattr(f, 'status'):  7
#         result['status'] = f.status
#     f.close()
#     return result
# 1	首先, 你用 URL, ETag hash, Last-Modified 日期 和 User-Agent 调用 openAnything 函数。
# 2	读取从服务器返回的真实数据。 这可能是被压缩的; 如果是, 将在后面进行解压缩。
# 3	保存从服务器返回的 ETag hash, 所以调用程序下一次能通过它返回给你, 并且可以传递给 openAnything, 连同 If-None-Match 头信息一起发送给远程服务器。
# 4	也要保存 Last-Modified 数据。
# 5	如果服务器说它发送的是压缩数据, 就执行解压缩。
# 6	如果你的服务器返回一个 URL 就保存它, 并在查明之前假定状态代码为 200。
# 7	如果其中一个自定义 URL 处理器捕获了一个状态代码, 也要保存下来。
# 例 11.19. 使用 openanything.py
# >>> import openanything
# >>> useragent = 'MyHTTPWebServicesApp/1.0'
# >>> url = 'http://diveintopython.org/redir/example301.xml'
# >>> params = openanything.fetch(url, agent=useragent)  1
# >>> params  2
# {'url': 'http://diveintomark.org/xml/atom.xml', 'lastmodified': 'Thu, 15 Apr 2004 19:45:21 GMT', 'etag': '"e842a-3e53-55d97640"', 'status': 301, 'data': '<?xml version="1.0" encoding="iso-8859-1"?> <feed version="0.3" <-- rest of data omitted for brevity -->'}
# >>> if params['status'] == 301:  3
# ... url = params['url']
# >>> newparams = openanything.fetch(
# ... url, params['etag'], params['lastmodified'], useragent)  4
# >>> newparams
# {'url': 'http://diveintomark.org/xml/atom.xml', 'lastmodified': None, 'etag': '"e842a-3e53-55d97640"', 'status': 304, 'data': ''}  5
# 1	真正第一次获取资源时, 你没有 ETag hash 或 Last-Modified 日期, 所以你不用使用这些参数。 (They're 可选参数。)
# 2	你获得了一个 dictionary, 它包括几个有用的头信息, HTTP 状态代码 和 从服务器返回的真实数据。 openanything 在内部处理 gzip 压缩; 在本级别上你不必关心它。
# 3	如果你得到一个 301 状态代码, 表示是个永久重定向, 你需要更新你的 URL 到新地址。
# 4	第二次获取相同的资源, 你已经从以往获得了各种信息: URL (可能被更新了), 从上一次访问获得的 ETag, 从上一次访问获得的 Last-Modified 日期, 当然还有 User-Agent。
# 5	你重新获取了这个 dictionary, 但是数据没有改变, 所以你得到了一个 304 状态代码而没有数据。