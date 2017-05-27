#/usr/lib/python3.5
# -*- coding: utf-8 -*-

#来源：http://www.cnblogs.com/xiaowuyi/archive/2012/10/15/2721658.html
#来源：http://blog.csdn.net/tianxicool/article/details/5942523


一、利用HTMLParser进行网页解析 
具体HTMLParser官方文档可参考http://docs.python.org/library/htmlparser.html#HTMLParser.HTMLParser

1、从一个简单的解析例子开始 
例1： 
test1.html文件内容如下：

复制代码
<html> 
<head> 
<title> XHTML 与 HTML 4.01 标准没有太多的不同</title> 
</head> 
<body> 
i love you 
</body> 
</html> 
复制代码
 

下面是能够列出title和body的程序示例：

复制代码
##@小五义：http://www.cnblogs.com/xiaowuyi 
##HTMLParser示例 
import HTMLParser 
class TitleParser(HTMLParser.HTMLParser): 
    def __init__(self): 
        self.taglevels=[] 
        self.handledtags=['title','body'] #提出标签 
        self.processing=None 
        HTMLParser.HTMLParser.__init__(self) 
    def handle_starttag(self,tag,attrs): 
        if tag in self.handledtags: 
            self.data='' 
            self.processing=tag 
    def handle_data(self,data): 
        if self.processing: 
            self.data +=data 
    def handle_endtag(self,tag): 
        if tag==self.processing: 
            print str(tag)+':'+str(tp.gettitle()) 
            self.processing=None 
    def gettitle(self): 
        return self.data 
fd=open('test1.html') 
tp=TitleParser() 
tp.feed(fd.read()) 
复制代码
 

运行结果如下： 
title: XHTML 与 HTML 4.01 标准没有太多的不同 
body: 
i love you 
程序定义了一个TitleParser类，它是HTMLParser类的子孙。HTMLParser的feed方法将接收数据，并通过定义的HTMLParser对象对数据进行相应的解析。其中handle_starttag、handle_endtag判断起始和终止tag，handle_data检查是否取得数据，如果self.processing不为None，那么就取得数据。

2、解决html实体问题 
（HTML 中有用的字符实体） 
（1）实体名称 
当与到HTML中的实体问题时，上面的例子就无法实现，如这里将test1.html的代码改为： 
例2：

复制代码
<html> 
<head> 
<title> XHTML 与&quot; HTML 4.01 &quot;标准没有太多的不同</title> 
</head> 
<body> 
i love you&times; 
</body> 
</html> 
复制代码
 

利用上面的例子进行分析，其结果是： 
title: XHTML 与 HTML 4.01 标准没有太多的不同 
body: 
i love you 
实体完全消失了。这是因为当出现实体的时候，HTMLParser调用了handle_entityref()方法，因为代码中没有定义这个方法，所以就什么都没有做。经过修改后，如下：

复制代码
##@小五义：http://www.cnblogs.com/xiaowuyi 
##HTMLParser示例：解决实体问题 
from htmlentitydefs import entitydefs 
import HTMLParser 
class TitleParser(HTMLParser.HTMLParser): 
    def __init__(self): 
        self.taglevels=[] 
        self.handledtags=['title','body'] 
        self.processing=None 
        HTMLParser.HTMLParser.__init__(self) 
    def handle_starttag(self,tag,attrs): 
        if tag in self.handledtags: 
            self.data='' 
            self.processing=tag 
    def handle_data(self,data): 
        if self.processing: 
            self.data +=data 
    def handle_endtag(self,tag): 
        if tag==self.processing: 
            print str(tag)+':'+str(tp.gettitle()) 
            self.processing=None 
    def handle_entityref(self,name): 
        if entitydefs.has_key(name): 
            self.handle_data(entitydefs[name]) 
        else: 
            self.handle_data('&'+name+';') 
    def gettitle(self): 
        return self.data 
fd=open('test1.html') 
tp=TitleParser() 
tp.feed(fd.read()) 
复制代码
 

运行结果为： 
title: XHTML 与" HTML 4.01 "标准没有太多的不同 
body: 
i love you× 
这里就把所有的实体显示出来了。

（2）实体编码 
例3：

复制代码
<html> 
<head> 
<title> XHTML 与&quot; HTML 4.01 &quot;标准没有太多的不同</title> 
</head> 
<body> 
i love&#247; you&times; 
</body> 
</html> 
复制代码
 

如果利用例2的代码执行后结果为： 
title: XHTML 与" HTML 4.01 "标准没有太多的不同 
body: 
i love you× 
结果中&#247; 对应的÷没有显示出来。 
添加handle_charref（）进行处理，具体代码如下：

复制代码
##@小五义：http://www.cnblogs.com/xiaowuyi 
##HTMLParser示例：解决实体问题 
from htmlentitydefs import entitydefs 
import HTMLParser 
class TitleParser(HTMLParser.HTMLParser): 
    def __init__(self): 
        self.taglevels=[] 
        self.handledtags=['title','body'] 
        self.processing=None 
        HTMLParser.HTMLParser.__init__(self) 
    def handle_starttag(self,tag,attrs): 
        if tag in self.handledtags: 
            self.data='' 
            self.processing=tag 
    def handle_data(self,data): 
        if self.processing: 
            self.data +=data 
    def handle_endtag(self,tag): 
        if tag==self.processing: 
            print str(tag)+':'+str(tp.gettitle()) 
            self.processing=None 
    def handle_entityref(self,name): 
        if entitydefs.has_key(name): 
            self.handle_data(entitydefs[name]) 
        else: 
            self.handle_data('&'+name+';') 

    def handle_charref(self,name): 
        try: 
            charnum=int(name) 
        except ValueError: 
            return 
        if charnum<1 or charnum>255: 
            return 
        self.handle_data(chr(charnum)) 

    def gettitle(self): 
        return self.data 
fd=open('test1.html') 
tp=TitleParser() 
tp.feed(fd.read()) 
复制代码
 

运行结果为： 
title: XHTML 与" HTML 4.01 "标准没有太多的不同 
body: 
i love÷ you×

3、提取链接 
例4： 

复制代码
<html> 
<head> 
<title> XHTML 与&quot; HTML 4.01 &quot;标准没有太多的不同</title> 
</head> 
<body> 

<a href="http://pypi.python.org/pypi" title="link1">i love&#247; you&times;</a> 
</body> 
</html>
复制代码
 


这里在handle_starttag(self,tag,attrs)中，tag=a时，attrs记录了属性值，因此只需要将attrs中name=href的value提出即可。具体如下：

复制代码
##@小五义：http://www.cnblogs.com/xiaowuyi 
##HTMLParser示例：提取链接 
# -*- coding: cp936 -*- 
from htmlentitydefs import entitydefs 
import HTMLParser 
class TitleParser(HTMLParser.HTMLParser): 
    def __init__(self): 
        self.taglevels=[] 
        self.handledtags=['title','body'] 
        self.processing=None 
        HTMLParser.HTMLParser.__init__(self)        
    def handle_starttag(self,tag,attrs): 
        if tag in self.handledtags: 
            self.data='' 
            self.processing=tag 
        if tag =='a': 
            for name,value in attrs: 
                if name=='href': 
                    print '连接地址：'+value 
    def handle_data(self,data): 
        if self.processing: 
            self.data +=data 
    def handle_endtag(self,tag): 
        if tag==self.processing: 
            print str(tag)+':'+str(tp.gettitle()) 
            self.processing=None 
    def handle_entityref(self,name): 
        if entitydefs.has_key(name): 
            self.handle_data(entitydefs[name]) 
        else: 
            self.handle_data('&'+name+';') 

    def handle_charref(self,name): 
        try: 
            charnum=int(name) 
        except ValueError: 
            return 
        if charnum<1 or charnum>255: 
            return 
        self.handle_data(chr(charnum)) 

    def gettitle(self): 
        return self.data 
fd=open('test1.html') 
tp=TitleParser() 
tp.feed(fd.read()) 
复制代码
 

运行结果为： 
title: XHTML 与" HTML 4.01 "标准没有太多的不同 
连接地址：http://pypi.python.org/pypi 
body:

i love÷ you×

4、提取图片 
如果网页中有一个图片文件，将其提取出来，并存为一个单独的文件。 
例5：

复制代码
<html> 
<head> 
<title> XHTML 与&quot; HTML 4.01 &quot;标准没有太多的不同</title> 
</head> 
<body> 
i love&#247; you&times; 
<a href="http://pypi.python.org/pypi" title="link1">我想你</a> 
<div id="m"><img src="http://www.baidu.com/img/baidu_sylogo1.gif" width="270" height="129" ></div> 
</body> 
</html> 
复制代码
 


将baidu_sylogo1.gif存取出来，具体代码如下：

复制代码
##@小五义：http://www.cnblogs.com/xiaowuyi 
##HTMLParser示例：提取图片 
# -*- coding: cp936 -*- 
from htmlentitydefs import entitydefs 
import HTMLParser,urllib 
def getimage(addr):#提取图片并存在当前目录下 
    u = urllib.urlopen(addr) 
    data = u.read() 
    filename=addr.split('/')[-1] 
    f=open(filename,'wb') 
    f.write(data) 
    f.close() 
    print filename+'已经生成！' 

class TitleParser(HTMLParser.HTMLParser): 
    def __init__(self): 
        self.taglevels=[] 
        self.handledtags=['title','body'] 
        self.processing=None 
        HTMLParser.HTMLParser.__init__(self)        
    def handle_starttag(self,tag,attrs): 
        if tag in self.handledtags: 
            self.data='' 
            self.processing=tag 
        if tag =='a': 
            for name,value in attrs: 
                if name=='href': 
                    print '连接地址：'+value 
        if tag=='img': 
            for name,value in attrs: 
                if name=='src': 
                    getimage(value) 
    def handle_data(self,data): 
        if self.processing: 
            self.data +=data 
    def handle_endtag(self,tag): 
        if tag==self.processing: 
            print str(tag)+':'+str(tp.gettitle()) 
            self.processing=None 
    def handle_entityref(self,name): 
        if entitydefs.has_key(name): 
            self.handle_data(entitydefs[name]) 
        else: 
            self.handle_data('&'+name+';') 

    def handle_charref(self,name): 
        try: 
            charnum=int(name) 
        except ValueError: 
            return 
        if charnum<1 or charnum>255: 
            return 
        self.handle_data(chr(charnum)) 

    def gettitle(self): 
        return self.data 
fd=open('test1.html') 
tp=TitleParser() 
tp.feed(fd.read()) 
复制代码
 

运动结果为： 
title: XHTML 与" HTML 4.01 "标准没有太多的不同 
连接地址：http://pypi.python.org/pypi 
baidu_sylogo1.gif已经生成！ 
body: 
i love÷ you× 
?ò????

5、实际例子： 
例6、获取人人网首页上的各各链接地址，代码如下：

复制代码
##@小五义：http://www.cnblogs.com/xiaowuyi 
##HTMLParser示例：获取人人网首页上的各各链接地址 
#coding: utf-8 
from htmlentitydefs import entitydefs 
import HTMLParser,urllib 
def getimage(addr): 
    u = urllib.urlopen(addr) 
    data = u.read() 
    filename=addr.split('/')[-1] 
    f=open(filename,'wb') 
    f.write(data) 
    f.close() 
    print filename+'已经生成！' 
class TitleParser(HTMLParser.HTMLParser): 
    def __init__(self): 
        self.taglevels=[] 
        self.handledtags=['a'] 
        self.processing=None 
        self.linkstring='' 
        self.linkaddr='' 
        HTMLParser.HTMLParser.__init__(self)        
    def handle_starttag(self,tag,attrs): 
        if tag in self.handledtags: 
            for name,value in attrs: 
                if name=='href': 
                    self.linkaddr=value 
            self.processing=tag 

    def handle_data(self,data): 
        if self.processing: 
            self.linkstring +=data 
            #print data.decode('utf-8')+':'+self.linkaddr 
    def handle_endtag(self,tag): 
        if tag==self.processing: 
            print self.linkstring.decode('utf-8')+':'+self.linkaddr 
            self.processing=None 
            self.linkstring='' 
    def handle_entityref(self,name): 
        if entitydefs.has_key(name): 
            self.handle_data(entitydefs[name]) 
        else: 
            self.handle_data('&'+name+';') 

    def handle_charref(self,name): 
        try: 
            charnum=int(name) 
        except ValueError: 
            return 
        if charnum<1 or charnum>255: 
            return 
        self.handle_data(chr(charnum)) 

    def gettitle(self): 
        return self.linkaddr 
tp=TitleParser() 
tp.feed(urllib.urlopen('http://www.renren.com/').read())
复制代码
 

运行结果： 
分享:http://share.renren.com 
应用程序:http://app.renren.com 
公共主页:http://page.renren.com 
人人生活:http://life.renren.com 
人人小组:http://xiaozu.renren.com/ 
同名同姓:http://name.renren.com 
人人中学:http://school.renren.com/allpages.html 
大学百科:http://school.renren.com/daxue/ 
人人热点:http://life.renren.com/hot 
人人小站:http://zhan.renren.com/ 
人人逛街:http://j.renren.com/ 
人人校招:http://xiaozhao.renren.com/ 
:http://www.renren.com 
注册:http://wwv.renren.com/xn.do?ss=10113&rt=27 
登录:http://www.renren.com/ 
帮助:http://support.renren.com/helpcenter 
给我们提建议:http://support.renren.com/link/suggest 
更多:# 
:javascript:closeError(); 
打开邮箱查收确认信:# 
重新输入:javascript:closeError(); 
:javascript:closeStop(); 
客服:http://help.renren.com/#http://help.renren.com/support/contomvice?pid=2&selection={couId:193,proId:342,cityId:1000375} 
:javascript:closeLock(); 
立即解锁:http://safe.renren.com/relive.do 
忘记密码？:http://safe.renren.com/findPass.do 
忘记密码？:http://safe.renren.com/findPass.do 
换一张:javascript:refreshCode_login(); 
MSN:# 
360:https://openapi.360.cn/oauth2/authorize?client_id=5ddda4458747126a583c5d58716bab4c&response_type=code&redirect_uri=http://www.renren.com/bind/tsz/tszLoginCallBack&scope=basic&display=default 
天翼:https://oauth.api.189.cn/emp/oauth2/authorize?app_id=296961050000000294&response_type=code&redirect_uri=http://www.renren.com/bind/ty/tyLoginCallBack 
为什么要填写我的生日？:#birthday 
看不清换一张?:javascript:refreshCode(); 
想了解更多人人网功能？点击此处:javascript:; 
:javascript:; 
:javascript:; 
立刻注册:http://reg.renren.com/xn6245.do?ss=10113&rt=27 
关于:http://www.renren.com/siteinfo/about 
开放平台:http://dev.renren.com 
人人游戏:http://wan.renren.com 
公共主页:http://page.renren.com/register/regGuide/ 
手机人人:http://mobile.renren.com/mobilelink.do?psf=40002 
团购:http://www.nuomi.com 
皆喜网:http://www.jiexi.com 
营销服务:http://ads.renren.com 
招聘:http://job.renren-inc.com/ 
客服帮助:http://support.renren.com/helpcenter 
隐私:http://www.renren.com/siteinfo/privacy 
京ICP证090254号:http://www.miibeian.gov.cn/ 
互联网药品信息服务资格证:http://a.xnimg.cn/n/core/res/certificate.jpg

二、利用BeautifulSoup进行网页解析 
1、BeautifulSoup下载和安装 
下载地址：http://www.crummy.com/software/BeautifulSoup/download/3.x/ 
中文文档地址：http://www.crummy.com/software/BeautifulSoup/bs3/documentation.zh.html#Entity%20Conversion 
安装方法：将下载的文件解压缩后，文件夹下有个setup.py文件，然后在cmd下，运行python setup.py install进行安装，注意setup.py的路径问题。安装成功后，在python中就可以直接import BeautifulSoup了。 
2、从一个简单的解析例子开始 
例7：

复制代码
<html> 
<head> 
<title> XHTML 与&quot; HTML 4.01 &quot;标准没有太多的不同</title> 
</head> 
<body> 
i love&#247; you&times; 
<a href="http://pypi.python.org/pypi" title="link1">我想你</a> 
<div id="m"><img src="http://www.baidu.com/img/baidu_sylogo1.gif" width="270" height="129" ></div> 
</body> 
</html> 
复制代码
 

获取title的代码：

复制代码
##@小五义：http://www.cnblogs.com/xiaowuyi 
##BeautifulSoup示例：title 
#coding: utf8 
import BeautifulSoup 

a=open('test1.html','r') 
htmlline=a.read() 
soup=BeautifulSoup.BeautifulSoup(htmlline.decode('gb2312')) 
#print soup.prettify()#规范化html文件 
titleTag=soup.html.head.title 
print titleTag.string 
复制代码
 

运行结果： 
XHTML 与&quot; HTML 4.01 &quot;标准没有太多的不同 
从代码和结果来看，应注意两点： 
第一，在BeautifulSoup.BeautifulSoup(htmlline.decode('gb2312'))初始化过程中，应注意字符编码格式，从网上搜索了一下，开始用utf-8的编码显示不正常，换为gb2312后显示正常。其实可以用soup.originalEncoding方法来查看原文件的编码格式。 
第二，结果中未对字符实体进行处理，在BeautifulSoup中文文档中，有专门对实体转换的解释，这里将上面的代码改为以下代码后，结果将正常显示：

复制代码
##@小五义：http://www.cnblogs.com/xiaowuyi 
##BeautifulSoup示例：title 
#coding: utf8 
import BeautifulSoup 
a=open('test1.html','r') 
htmlline=a.read() 
soup=BeautifulSoup.BeautifulStoneSoup(htmlline.decode('gb2312'),convertEntities=BeautifulSoup.BeautifulStoneSoup.ALL_ENTITIES) 
#print soup.prettify()#规范化html文件 
titleTag=soup.html.head.title 
print titleTag.string 
复制代码
 

这里convertEntities=BeautifulSoup.BeautifulStoneSoup.ALL_ENTITIES中的ALL_ENTITIES定义了XML和HTML两者的实体代码。当然，也可以直接用XML_ENTITIES或者HTML_ENTITIES。运行结果如下： 
XHTML 与" HTML 4.01 "标准没有太多的不同 
3、提取链接 
还有用上面的例子，这里代码变为：

复制代码
##@小五义：http://www.cnblogs.com/xiaowuyi 
##BeautifulSoup示例：提取链接 
#coding: utf8 
import BeautifulSoup 
a=open('test1.html','r') 
htmlline=a.read() 
a.close() 
soup=BeautifulSoup.BeautifulStoneSoup(htmlline.decode('gb2312'),convertEntities=BeautifulSoup.BeautifulStoneSoup.ALL_ENTITIES) 
name=soup.find('a').string 
links=soup.find('a')['href'] 
print name+':'+links 
复制代码
 

运行结果为： 
我想你:http://pypi.python.org/pypi 
4、提取图片 
依然是用上面的例子，把baidu图片提取出来。 
代码为：

复制代码
##@小五义：http://www.cnblogs.com/xiaowuyi
#coding: utf8 
import BeautifulSoup,urllib 
def getimage(addr):#提取图片并存在当前目录下 
    u = urllib.urlopen(addr) 
    data = u.read() 
    filename=addr.split('/')[-1] 
    f=open(filename,'wb') 
    f.write(data) 
    f.close() 
    print filename+' finished!' 
a=open('test1.html','r') 
htmlline=a.read() 
soup=BeautifulSoup.BeautifulStoneSoup(htmlline.decode('gb2312'),convertEntities=BeautifulSoup.BeautifulStoneSoup.ALL_ENTITIES) 
links=soup.find('img')['src'] 
getimage(links) 
复制代码
 

提取链接和提取图片两部分主要都是用了find方法，具体方法为： 
find(name, attrs, recursive, text, **kwargs) 
findAll是列出全部符合条件的，find只列出第一条。这里注意的是findAll返回的是个list。 
5、实际例子： 
例8、获取人人网首页上的各各链接地址，代码如下：

复制代码
##@小五义：http://www.cnblogs.com/xiaowuyi 
##BeautifulSoup示例：获取人人网首页上的各各链接地址 
#coding: utf8 
import BeautifulSoup,urllib 
linkname='' 
htmlline=urllib.urlopen('http://www.renren.com/').read() 
soup=BeautifulSoup.BeautifulStoneSoup(htmlline.decode('utf-8')) 
links=soup.findAll('a') 
for i in links: 
    ##判断tag是a的里面，href是否存在。 
    if 'href' in str(i): 
        linkname=i.string 
        linkaddr=i['href'] 
        if 'NoneType' in str(type(linkname)):#当i无内容是linkname为Nonetype类型。 
            print linkaddr 
        else: 
            print linkname+':'+linkaddr 
复制代码
 

运行结果： 
分享:http://share.renren.com 
应用程序:http://app.renren.com 
公共主页:http://page.renren.com 
人人生活:http://life.renren.com 
人人小组:http://xiaozu.renren.com/ 
同名同姓:http://name.renren.com 
人人中学:http://school.renren.com/allpages.html 
大学百科:http://school.renren.com/daxue/ 
人人热点:http://life.renren.com/hot 
人人小站:http://zhan.renren.com/ 
人人逛街:http://j.renren.com/ 
人人校招:http://xiaozhao.renren.com/ 
http://www.renren.com 
注册:http://wwv.renren.com/xn.do?ss=10113&rt=27 
登录:http://www.renren.com/ 
帮助:http://support.renren.com/helpcenter 
给我们提建议:http://support.renren.com/link/suggest 
更多:# 
javascript:closeError(); 
打开邮箱查收确认信:# 
重新输入:javascript:closeError(); 
javascript:closeStop(); 
客服:http://help.renren.com/#http://help.renren.com/support/contomvice?pid=2&selection={couId:193,proId:342,cityId:1000375} 
javascript:closeLock(); 
立即解锁:http://safe.renren.com/relive.do 
忘记密码？:http://safe.renren.com/findPass.do 
忘记密码？:http://safe.renren.com/findPass.do 
换一张:javascript:refreshCode_login(); 
MSN:# 
360:https://openapi.360.cn/oauth2/authorize?client_id=5ddda4458747126a583c5d58716bab4c&response_type=code&redirect_uri=http://www.renren.com/bind/tsz/tszLoginCallBack&scope=basic&display=default 
天翼:https://oauth.api.189.cn/emp/oauth2/authorize?app_id=296961050000000294&response_type=code&redirect_uri=http://www.renren.com/bind/ty/tyLoginCallBack 
#birthday 
看不清换一张?:javascript:refreshCode(); 
javascript:; 
javascript:; 
立刻注册:http://reg.renren.com/xn6245.do?ss=10113&rt=27 
关于:http://www.renren.com/siteinfo/about 
开放平台:http://dev.renren.com 
人人游戏:http://wan.renren.com 
公共主页:http://page.renren.com/register/regGuide/ 
手机人人:http://mobile.renren.com/mobilelink.do?psf=40002 
团购:http://www.nuomi.com 
皆喜网:http://www.jiexi.com 
营销服务:http://ads.renren.com 
招聘:http://job.renren-inc.com/ 
客服帮助:http://support.renren.com/helpcenter 
隐私:http://www.renren.com/siteinfo/privacy 
京ICP证090254号:http://www.miibeian.gov.cn/ 
互联网药品信息服务资格证:http://a.xnimg.cn/n/core/res/certificate.jpg

分类: python学习笔记, python网络编程























20.1 HTMLParser- 简单的 HTML 和 XHTML parser

 

在 Python 3.0 中， HTMLParser is renamed to html.parser

 

该模块定义了一个类 HTMLParser ，用来解析 HTML 文本文件，也包括 XHTML 。和 htmllib 不同，它并非基于 SGMLparser 。

 

class HTMLParser.HTMLParser

使用 HTMLParser 的实例，填充 HTML 数据，并在开始和结束标记间调用函数。 HTMLParser 类意味着重载。

 

和 htmllib 的分析器不同， this parser 并不检测和开始标记对应的结束标记 or call the end-tag handler for elements which are close implicitly by closing an outer element.

 

这里还有一个例外情况：

exception HTMLParser.HTMLParserError

当分析遇到 Error 时 HTMLParser 会抛出异常。该异常提供三个属性： msg ， lineno and offset 。

 

HTMLParser 实例有如下的方法：

 

HTMLParser.reset()

重置实例 . 所有未处理的数据都会丢失。在初始化时自动调用。

 

HTMLParser.feed(data)

给分析器喂食。在由完整元素构成的情况下工作；不完整数据情况下，会进行缓冲知道更多数据加进来或者 close() 被调用。

 

HTMLParser.close()

处理所有缓冲数据。这个方法可以被派生类重定义，以便在输入结束后处理额外的事情，重定义的版本也要调用 HTMLParser 基类的 close() 方法。

 

HTMLParser.getpos()

返回当前行数和列数

 

HTMLParser.get_starttag_text()

返回最近打开过得开始标记处的文本。通常不会用到， but may be useful in dealing with HTML “as deployed” or for re-generating input with minimal changes (whitespace between attributes can be preserved, etc.).

 

HTMLParser.handle_starttag(tag, attrs)

该方法用来处理一个标记的开始。通常被派生类重载；基类的实例什么都不做。

tag 参数是 tag 的名字的小写化。 attrs 参数是一个 list ，由 (name, value) 组成，反映了 <> 里面的属性。 name 会被翻译成小写字母，在 value 中的引号也被移除了，字符实体引用也会被替换。例如，有个 tag<A HREF=”http://www.cwi.nl/”> ，那么使用该方法就该这么做： handle_starttag(’a’, [(’href’, ’http://www.cwi.nl/’)])

 

Changed in version 2.6: 来自 htmlentitydefs 的所有实体引用都被属性值替换。

 

HTMLParser.handle_startendtag(tag, attrs)

和 handle_starttag() 类似，用来处理 XHTML 风格的 空标签（ <a .../> ）。可能被子类重载， which require this particular lexical information; 默认的实现只是简单的调用 handle_starttag() 和 handle_endtag()

 

HTMLParser.handle_endtag(tag)

该方法用来处理元素结束标记。可以被派生类重载；基类什么也不做。 tag 参数是 tag 的 name 转化来的小写字母。

 

HTMLParser.handle_data(data)

       该方法用来处理随机的数据。可以被派生类重载；基类什么也不做。

 

HTMLParser.handle_charref(name)

       处理 &#ref 格式的字符引用。可以被派生类重载；基类什么也不做。

 

HTMLParser.handle_entityref(name)

       处理一般的 &name 格式的实体引用。 name 是一个一般的实体引用。可以被派生类重载；基类什么也不做。

 

HTMLParser.handle_comment(data)

处理遇到注释的情况。注释参数为在——和——之间的字符串文本，而不是分隔符自身。例如 <!--text--> ，该方法将调用‘ text ’。可以被派生类重载；基类什么也不做。

 

HTMLParser.handle_decl(decl)

当分析器遇到 SGML 声明时调用此方法。 decl 参数是 <!...> 标记里的整个内容。可以被派生类重载；基类什么也不做。

 

 

HTMLParser.handle_pi(data)

处理命令， data 参数包含整个的处理命令。例如 <?proc color=’red’> ，该方法应写成 handle_pi(”proc color=’red’”). 可以被派生类重载；基类什么也不做。



