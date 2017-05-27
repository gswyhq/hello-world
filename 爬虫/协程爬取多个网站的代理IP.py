# -*- coding: utf-8 -*-


#程序运行时间为: 5.789090709508604
#但有时候会卡在line=yield from reader.readline()不继续运行。
#http://blog.jobbole.com/63897/
#http://www.pythontip.com/blog/post/12142/
#http://www.oschina.net/translate/gevent-asynchronous-io-made-easy
#http://segmentfault.com/a/1190000000471602

#可以获取代理IP地址的网站
#http://www.my-proxy.com
#http://www.proxylists.net
#http://proxylist.sakura.ne.jp
#http://www.my-proxy.com
#http://www.cybersyndrome.net
#http://cn-proxy.com
#http://www.youdaili.cn
#http://www.itmop.com
#http://pachong.org
#http://www.cz88.net
#http://www.kuaidaili.com/
#http://www.xicidaili.com/

#协程爬取多个网站的代理IP
import asyncio,time
from bs4 import BeautifulSoup
import threading


@asyncio.coroutine
#把一个生成器generator标记为协程coroutine类型，然后，我们就把这个协程coroutine扔到条件循环EventLoop中执行。
def wget(host,uri):
    #print('%s' % host+uri)
    connect = asyncio.open_connection(host, 80)
    #asyncio.open_connection(host=None, port=None, *, loop=None, limit=None, **kwds)
    #包装了一个create_connection()返回一个（reader,writer)对。
    #返回的reader是一个StreamReader实例；writer是一个StreamWriter的实例。
    #这个函数是一个协程。
    reader, writer = yield from connect
    header = 'GET %s HTTP/1.0\r\nHost: %s\r\n\r\n' %(uri,host)
    print("A当前线程数是：",threading.active_count())
    #请求报文是由请求方法、请求 URI、协议版本、可选的请求首部字段和内容实体构成的。
    #起始行开头的GET表示请求访问服务器的类型，称为方法（method）。
    #随后的字符串 /  指明了请求访问的资源对象，也叫做请求 URI（request-URI）。
    #RUI可以是完整的请求URI，当然也可以把网络域名或IP地址写在首部字段的Host中。
    #最后的 HTTP/1.1，即 HTTP 的版本号，用来提示客户端使用的 HTTP 协议功能。
    #综合来看，这段请求内容的意思是：请求访问某台（host对应的） HTTP 服务器上的 / 页面资源。
    #与 URI（统一资源标识符）相比，我们更熟悉 URL（Uniform Resource Locator，统一资源定位符）。
    #URI 就是由某个协议方案表示的资源的定位标识符。协议方案是指访问资源所使用的协议类型名称。
    #采用 HTTP 协议时，协议方案就是 http。除此之外，还有 ftp、mailto、telnet、file 等。
    #URI 用字符串标识某一互联网资源，而 URL 表示资源的地点（互联网上所处的位置）。可见 URL 是 URI 的子集。
    #HTTP 报文本身是由多行（用 CR+LF 作换行符）数据构成的字符串文本。
    # 回车CR-将光标移动到当前行的开头。换行LF-将光标“垂直”移动到下一行。（而并不移动到下一行的开头，即不改变光标水平位置）
    #CR用符号'\r'表示, 十进制ASCII代码是13, 十六进制代码为0x0D；LF使用'\n'符号表示, ASCII代码是10, 十六制为0x0A
    writer.write(header.encode('utf-8'))
    #writer.write(data),写一些数据data传输;此方法不会阻塞，以异步的方式发生缓冲数据。
    yield from writer.drain()
    #writer.drain()，刷新缓冲区。其本身也是一个协程。
    asyncio.sleep(1)
    try:
        #source = yield from reader.read()
        print(1)
        source=b''
        num=0
        start1=time.clock()
        while 1:
            line=yield from reader.readline()
            #print(line)
            source+=line
            num+=1
            if not line:
                break
        #writer.close()
        asyncio.sleep(0.1)
        end1=time.clock()
        print("while时间为:",(end1-start1))
        print("num:",num)
        print("uri:",uri)
        print("当前线程数是：",threading.active_count())
        print(2)
        #ReadLine（）读取一行，其中“行”是\n结束字节序列。
        #如果EOF被接收，并且\ n未发现，该方法将返回部分读取字节。
        #如果EOF被接收和内部缓冲区是空的，返回一个空字节对象。
        #这种方法是一个协同程序。
        source=source.decode('utf-8')
    #print('网页内容：',source)
        soup = BeautifulSoup(source)
    #soup得到BeautifulSoup处理格式化后的字符串
    #print(soup.prettify())

        #响应报文基本上由协议版本、状态码（表示请求成功或失败的数字代码）、用以解释状态码的原因短语、可选的响应首部字段以及实体主体构成。
        #起始行开头的 HTTP/1.1 表示服务器对应的 HTTP 版本。紧挨着的 200 OK 表示请求的处理结果的状态码（status code）和
        #原因短语（reason-phrase）。下一行显示了创建响应的日期时间，是首部字段（header field）内的一个属性。
        #接着以一空行分隔，之后的内容称为资源实体的主体（entity body）。
        #不同网站的可选的响应首部字段不一样。
        list_soup = soup.find('tbody')#通过标签进行查找,找到标签tbody下的所有内容。
    except Exception as e:
        print("程序运行出错。++++++++++++++++++++=.%s"%(e))
    proxys=[]
    try:
        print(3)
        for tr_info in list_soup.find_all('tr'):#遍历tbody标签下的所有tr标签。
            for td_info in tr_info.find_all('td'):#遍历tr标签下的所以td标签。
                if td_info.find('span', {'class':'row_proxy_ip'}):#根据标签及属性查找到代理ip的标签及内容
                    ip = td_info.find('span', {'class':'row_proxy_ip'}).string.strip()
                #string属性，如果超过一个标签的话，那么就会返回None，否则就返回具体的字符串
                    port_info=td_info.find_next_sibling('td')#查找紧接着代理ip标签的兄弟标签td
                    ports=port_info.strings
                #strings属性，如果超过一个标签的话，也会返回具体的字符串的生成器generator object。若标签内容为空，也可能会返回\n等内容。
                    for p in ports:#遍历生成器
                        port=p.strip()#除去前后的空格、\n、\r等内容。
                        if port:
                            break #若找到代理IP端口了，就跳过。
                    break #若代理ip、端口号都找到了，就挑出。

                """
                这段也可以用下面这段代码代替
                #print(port_info)
                if port_info.string:
                    port=port_info.string
                else:
                    port=port_info.find('a').string
                if port:
                    port=port.strip()
            #re.search('\d{2,5}',desc)

                """
            proxys.append([ip,port])
        #s.strip(rm)        删除s字符串中开头、结尾处，位于 rm删除序列的字符;当rm为空时，默认删除空白符（包括'\n', '\r',  '\t',  ' ')
    except Exception as e:
        print("程序运行出错。%s===========.%s"%(ip,e))

    a=time.asctime()
    file_txt="F:/python/git/"+"爬取的代理服务器ip "+a[-4:]+a[3:10]+".txt"
    f= open(file_txt,'a+')
    #print(len(proxys))
    for proxy in proxys:
        f.write(r"http://%s:%s\n"%(proxy[0],proxy[1]))
    f.close()

    writer.close()
    #关闭传输
    #如果传输具有用于输出数据的缓冲器，缓冲的数据将被异步刷新。没有更多的数据将被接收。
#"""

start=time.clock()
loop = asyncio.get_event_loop()
#得到一个标准的事件循环EventLoop
#相当于调用get_event_loop_policy().get_event_loop()；
#asyncio.get_event_loop_policy（）获取当前事件循环的政策。
#get_event_loop（）获取当前事件循环
#"""
urls=[('www.proxynova.com', '/proxy-server-list/country-cn/'),
      ('www.proxynova.com', '/proxy-server-list/country-ve/'),
      ('www.proxynova.com', '/proxy-server-list/country-tw/'),
      ('www.proxynova.com', '/proxy-server-list/country-in/'),
      ('www.proxynova.com', '/proxy-server-list/country-br/'),
      ('www.proxynova.com', '/proxy-server-list/country-id/'),
      ('www.proxynova.com', '/proxy-server-list/country-us/'),
      ('www.proxynova.com', '/proxy-server-list/country-th/'),
      ('www.proxynova.com', '/proxy-server-list/country-ru/'),
      ('www.proxynova.com', '/proxy-server-list/country-ar/'),
      ('www.proxynova.com', '/proxy-server-list/country-hk/'),
      ('www.proxynova.com', '/proxy-server-list/country-de/'),
      ('www.proxynova.com', '/proxy-server-list/country-eg/'),
      ('www.proxynova.com', '/proxy-server-list/country-jp/'),
      ('www.proxynova.com', '/proxy-server-list/country-cl/'),
      ('www.proxynova.com', '/proxy-server-list/country-ma/'),
      ('www.proxynova.com', '/proxy-server-list/country-co/'),
      ('www.proxynova.com', '/proxy-server-list/country-vn/'),
      ('www.proxynova.com', '/proxy-server-list/country-bd/'),
      ('www.proxynova.com', '/proxy-server-list/country-ua/'),
      ('www.proxynova.com', '/proxy-server-list/country-gb/'),
      ('www.proxynova.com', '/proxy-server-list/country-pl/'),
      ('www.proxynova.com', '/proxy-server-list/country-my/'),
      ('www.proxynova.com', '/proxy-server-list/country-fr/')]

    #"""
#urls=[('www.proxynova.com', '/proxy-server-list/country-cn/')]
tasks = [wget(host,uri) for host,uri in urls]
print("B当前线程数是：",threading.active_count())
loop.run_until_complete(asyncio.wait(tasks,timeout=20))
#设置超时为20秒，若删除超时，可能出现程序一直阻塞在某处不前行。
print("C当前线程数是：",threading.active_count())
#run_until_complete()方法来运行协同程序
#asyncio.wait(tasks),等待tasks提供的序列协程完成。
loop.close()
#"""
end=time.clock()
print("程序运行时间为:",(end-start))
print('程序运行完毕。')

"""
HTTP请求支持的方法
GET   请求获取Request-URI所标识的资源
POST 在Request-URI所标识的资源后附加新的数据
HEAD 请求获取由Request-URI所标识的资源的响应消息报头
PUT    请求服务器存储一个资源，并用Request-URI作为其标识
DELETE 请求服务器删除Request-URI所标识的资源
OPTIONS 请求查询服务器的性能，或者查询与资源相关的选项和需求
TRACE   请求服务器回送收到的请求信息，主要用于测试或诊断
CONNECT 隧道机制

HTTP的状态响应码
1XX:指示信息，请求收到，继续处理
2XX:成功，操作成功收到，分析，接受
3XX:完成请求必须进一步处理，重定向
4XX:请求包含一个错误语法，不能完成。指示客户端错误
5XX:服务器执行一个有效请求失败，指示服务端错误

其他HTTP重要的知识就是报头信息中的具体字段信息：
报头字段的格式为 （名字）“：”（空格）（值）；
通用报头字段：
Cache-Control
Date
Connection
请求报头字段：
Accept
Accept-Charset
Accept-Encoding
Accept-Language
Authorization
Host
User-Agent
响应报头字段：
Location
Server
实体报头字段：
Content-Encoding
Content-Language
Content-Length
Content-Type
Last-Modified
Expires


soup 就是BeautifulSoup处理格式化后的字符串，soup.title 得到的是title标签，soup.p  得到的是文档中的第一个p标签，要想得到所有标签，得用find_all函数。
find_all 函数返回的是一个序列，可以对它进行循环，依次得到想到的东西.
get_text() 是返回文本,这个对每一个BeautifulSoup处理后的对象得到的标签都是生效的。你可以试试 print soup.p.get_text()
其实是可以获得标签的其他属性的，比如我要获得a标签的href属性的值，可以使用 print soup.a['href'],类似的其他属性，
比如class也是可以这么得到的（soup.a['class']）。
特别的，一些特殊的标签，比如head标签，是可以通过soup.head 得到，其实前面也已经说了。
如何获得标签的内容数组？使用contents 属性就可以 比如使用 print soup.head.contents，就获得了head下的所有子孩子，以列表的形式返回结果，
可以使用 [num]  的形式获得 ,获得标签，使用.name 就可以。
获取标签的孩子，也可以使用children，但是不能print soup.head.children 没有返回列表，返回的是 <listiterator object at 0x108e6d150>,
不过使用list可以将其转化为列表。当然可以使用for 语句遍历里面的孩子。
关于string属性，如果超过一个标签的话，那么就会返回None，否则就返回具体的字符串
超过一个标签的话，可以试用strings；
向上查找可以用parent函数，如果查找所有的，那么可以使用parents函数；
查找下一个兄弟使用next_sibling,查找上一个兄弟节点使用previous_sibling,如果是查找所有的，那么在对应的函数后面加s就可以

"""
