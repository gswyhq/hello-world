# -*- coding: utf-8 -*-
'''
Created on 2011-11-11

@author: PaulWang

Description:


来源：http://blog.csdn.net/mhmyqn/article/details/25561535
HTTP请求中，如果是get请求，那么表单参数以name=value&name1=value1的形式附到url的后面，如果是post请求，那么表单参数是在请求体中，也是以name=value&name1=value1的形式在请求体中。通过chrome的开发者工具可以看到如下（这里是可读的形式，不是真正的HTTP请求协议的请求格式）：

get请求：

[plain] view plaincopy在CODE上查看代码片派生到我的代码片
RequestURL:http://127.0.0.1:8080/test/test.do?name=mikan&address=street  
Request Method:GET  
Status Code:200 OK  
   
Request Headers  
Accept:text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8  
Accept-Encoding:gzip,deflate,sdch  
Accept-Language:zh-CN,zh;q=0.8,en;q=0.6  
AlexaToolbar-ALX_NS_PH:AlexaToolbar/alxg-3.2  
Connection:keep-alive  
Cookie:JSESSIONID=74AC93F9F572980B6FC10474CD8EDD8D  
Host:127.0.0.1:8080  
Referer:http://127.0.0.1:8080/test/index.jsp  
User-Agent:Mozilla/5.0 (Windows NT 6.1)AppleWebKit/537.36 (KHTML, like Gecko) Chrome/33.0.1750.149 Safari/537.36  
   
Query String Parameters  
name:mikan  
address:street  
   
Response Headers  
Content-Length:2  
Date:Sun, 11 May 2014 10:42:38 GMT  
Server:Apache-Coyote/1.1  
Post请求：

[plain] view plaincopy在CODE上查看代码片派生到我的代码片
RequestURL:http://127.0.0.1:8080/test/test.do  
Request Method:POST  
Status Code:200 OK  
   
Request Headers  
Accept:text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8  
Accept-Encoding:gzip,deflate,sdch  
Accept-Language:zh-CN,zh;q=0.8,en;q=0.6  
AlexaToolbar-ALX_NS_PH:AlexaToolbar/alxg-3.2  
Cache-Control:max-age=0  
Connection:keep-alive  
Content-Length:25  
Content-Type:application/x-www-form-urlencoded  
Cookie:JSESSIONID=74AC93F9F572980B6FC10474CD8EDD8D  
Host:127.0.0.1:8080  
Origin:http://127.0.0.1:8080  
Referer:http://127.0.0.1:8080/test/index.jsp  
User-Agent:Mozilla/5.0 (Windows NT 6.1)AppleWebKit/537.36 (KHTML, like Gecko) Chrome/33.0.1750.149 Safari/537.36  
   
Form Data  
name:mikan  
address:street  
   
Response Headers  
Content-Length:2  
Date:Sun, 11 May 2014 11:05:33 GMT  
Server:Apache-Coyote/1.1  
这里要注意post请求的Content-Type为application/x-www-form-urlencoded，参数是在请求体中，即上面请求中的Form Data。

 在servlet中，可以通过request.getParameter(name)的形式来获取表单参数。

 而如果使用原生AJAX POST请求的话：

[javascript] view plaincopy在CODE上查看代码片派生到我的代码片
function getXMLHttpRequest() {  
          var xhr;  
          if(window.ActiveXObject) {  
                   xhr= new ActiveXObject("Microsoft.XMLHTTP");  
          }else if (window.XMLHttpRequest) {  
                   xhr= new XMLHttpRequest();  
          }else {  
                   xhr= null;  
          }  
          return xhr;  
}  
  
function save() {  
          var xhr = getXMLHttpRequest();  
          xhr.open("post","http://127.0.0.1:8080/test/test.do");  
          var data = "name=mikan&address=street...";  
          xhr.send(data);  
          xhr.onreadystatechange= function() {  
                   if(xhr.readyState == 4 && xhr.status == 200) {  
                            alert("returned:"+ xhr.responseText);  
                   }  
          };  
}  
 

通过chrome的开发者工具看到请求头如下：

[plain] view plaincopy在CODE上查看代码片派生到我的代码片
RequestURL:http://127.0.0.1:8080/test/test.do  
Request Method:POST  
Status Code:200 OK  
   
Request Headers  
Accept:*/*  
Accept-Encoding:gzip,deflate,sdch  
Accept-Language:zh-CN,zh;q=0.8,en;q=0.6  
AlexaToolbar-ALX_NS_PH:AlexaToolbar/alxg-3.2  
Connection:keep-alive  
Content-Length:28  
Content-Type:text/plain;charset=UTF-8  
Cookie:JSESSIONID=C40C7823648E952E7C6F7D2E687A0A89  
Host:127.0.0.1:8080  
Origin:http://127.0.0.1:8080  
Referer:http://127.0.0.1:8080/test/index.jsp  
User-Agent:Mozilla/5.0 (Windows NT 6.1)AppleWebKit/537.36 (KHTML, like Gecko) Chrome/33.0.1750.149 Safari/537.36  
   
Request Payload  
name=mikan&address=street  
   
Response Headers  
Content-Length:2  
Date:Sun, 11 May 2014 11:49:23 GMT  
Server:Apache-Coyote/1.1  
注意请求的Content-Type为text/plain;charset=UTF-8，而请求表单参数在RequestPayload中。

 那么servlet中通过request.getParameter(name)却是空。为什么呢？而这样的参数又该怎么样获取呢？

为了搞明白这个问题，查了些资料，也看了Tomcat7.0.53关于请求参数处理的源码，终于搞明白了是怎么回事。

HTTP POST表单请求提交时，使用的Content-Type是application/x-www-form-urlencoded，而使用原生AJAX的POST请求如果不指定请求头RequestHeader，默认使用的Content-Type是text/plain;charset=UTF-8。

 由于Tomcat对于Content-Type multipart/form-data（文件上传）和application/x-www-form-urlencoded（POST请求）做了“特殊处理”。下面来看看相关的处理代码。

Tomcat的HttpServletRequest类的实现类为org.apache.catalina.connector.Request（实际上是org.apache.coyote.Request），而它对处理请求参数的方法为protected void parseParameters()，这个方法中对Content-Type multipart/form-data（文件上传）和application/x-www-form-urlencoded（POST请求）的处理代码如下：


[java] view plaincopy在CODE上查看代码片派生到我的代码片
protectedvoid parseParameters() {  
           //省略部分代码......  
           parameters.handleQueryParameters();// 这里是处理url中的参数  
           //省略部分代码......  
           if ("multipart/form-data".equals(contentType)) { // 这里是处理文件上传请求  
                parseParts();  
                success = true;  
                return;  
           }  
   
           if(!("application/x-www-form-urlencoded".equals(contentType))) {// 这里如果是非POST请求直接返回，不再进行处理  
                success = true;  
                return;  
           }  
           //下面的代码才是处理POST请求参数  
           //省略部分代码......  
           try {  
                if (readPostBody(formData, len)!= len) { // 读取请求体数据  
                    return;  
                }  
           } catch (IOException e) {  
                // Client disconnect  
                if(context.getLogger().isDebugEnabled()) {  
                    context.getLogger().debug(  
                            sm.getString("coyoteRequest.parseParameters"),e);  
                }  
                return;  
           }  
           parameters.processParameters(formData, 0, len); // 处理POST请求参数，把它放到requestparameter map中（即request.getParameterMap获取到的Map，request.getParameter(name)也是从这个Map中获取的）  
           // 省略部分代码......  
}  
   
   protected int readPostBody(byte body[], int len)  
       throws IOException {  
   
       int offset = 0;  
       do {  
           int inputLen = getStream().read(body, offset, len - offset);  
           if (inputLen <= 0) {  
                return offset;  
           }  
           offset += inputLen;  
       } while ((len - offset) > 0);  
       return len;  
    }  

从上面代码可以看出，Content-Type不是application/x-www-form-urlencoded的POST请求是不会读取请求体数据和进行相应的参数处理的，即不会解析表单数据来放到request parameter map中。所以通过request.getParameter(name)是获取不到的。
 那么这样提交的参数我们该怎么获取呢？

当然是使用最原始的方式，读取输入流来获取了，如下所示：

[java] view plaincopy在CODE上查看代码片派生到我的代码片
privateString getRequestPayload(HttpServletRequest req) {  
          StringBuildersb = new StringBuilder();  
          try(BufferedReaderreader = req.getReader();) {  
                   char[]buff = new char[1024];  
                   intlen;  
                   while((len = reader.read(buff)) != -1) {  
                            sb.append(buff,0, len);  
                   }  
          }catch (IOException e) {  
                   e.printStackTrace();  
          }  
          returnsb.toString();  
}  
当然，设置了application/x-www-form-urlencoded的POST请求也可以通过这种方式来获取。

 所以，在使用原生AJAX POST请求时，需要明确设置Request Header，即：

[javascript] view plaincopy在CODE上查看代码片派生到我的代码片
xhr.setRequestHeader("Content-Type","application/x-www-form-urlencoded");  
另外，如果使用jquery，我使用1.11.0这个版本来测试，$.ajax post请求是不需要明确设置这个请求头的，其他版本的本人没有亲自测试过。相信在1.11.0之后的版本也是不需要设置的。不过之前有的就不一定了。这个没有测试过。


2015-04-17后记：

最近在看书时才真正搞明白，服务器为什么会对表单提交和文件上传做特殊处理，因为表单提交数据是名值对的方式，且Content-Type为application/x-www-form-urlencoded，而文件上传服务器需要特殊处理，普通的post请求（Content-Type不是application/x-www-form-urlencoded）数据格式不固定，不一定是名值对的方式，所以服务器无法知道具体的处理方式，所以只能通过获取原始数据流的方式来进行解析。

jquery在执行post请求时，会设置Content-Type为application/x-www-form-urlencoded，所以服务器能够正确解析，而使用原生ajax请求时，如果不显示的设置Content-Type，那么默认是text/plain，这时服务器就不知道怎么解析数据了，所以才只能通过获取原始数据流的方式来进行解析请求数据。
'''
#import urllib.request,urllib.parse
#
#import http.client
#
#params = urllib.parse.urlencode({'@email': '112233@gmail.com', '@password': '1212123', '@action': 'https://system.netsuite.com/app/login/nllogin.nl'})
#利用Urllib进行表单数据提交(Get,Post)

#headers = {"Content-type": "application/x-www-form-urlencoded"}
#来源：http://blog.csdn.net/serverxp/article/details/6963059
#
#conn = http.client.HTTPConnection("www.netsuite.com")
#conn.request("POST", "",params,headers)
#r1 = conn.getresponse()
#print(r1.status, r1.reason)
#data = r1.read()
#print(data)
#conn.close()

import urllib
import sys
import http.cookiejar


cookie = http.cookiejar.CookieJar()                                        #保存cookie，为登录后访问其它页面做准备
cjhdr  =  urllib.request.HTTPCookieProcessor(cookie)             
opener = urllib.request.build_opener(cjhdr)

#定义一个要提交的数据数组(字典)
data = {}
data['email'] = 'yicui49@gmail.com'
data['password'] = 'fashlets123'
data['Submit']=''

url = "https://system.netsuite.com/pages/customerlogin.jsp?country=US"
postdata = urllib.parse.urlencode(data)
postdata = postdata.encode('utf-8')

res = urllib.request.urlopen(url,postdata)
print(res.status, res.reason)

#获取提交后返回的信息
content = res.read()
print('*'*50,'\n',content)

if( res.status != 200 ):
    exit()

print('ok')




