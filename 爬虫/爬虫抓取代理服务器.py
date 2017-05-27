# -*- coding: utf-8 -*-
#爬虫抓取代理服务器
#http://f.dataguru.cn/thread-54108-1-1.html
import urllib.request
import re
import sys
import chardet
import threading
import time,socket
#reload(sys)
#sys.setdefaultencoding('utf-8') 

rawProxyList = []
checkedProxyList = []

#八个目标网站
targets = ['http://www.proxy360.cn/Region/Brazil',
           'http://www.proxy360.cn/Region/China',
           'http://www.proxy360.cn/Region/America',
           'http://www.proxy360.cn/Region/Taiwan',
           'http://www.proxy360.cn/Region/Japan',
           'http://www.proxy360.cn/Region/Thailand',
           'http://www.proxy360.cn/Region/Vietnam',
           'http://www.proxy360.cn/Region/bahrein'
           ]

#http://www.proxynova.com/proxy-server-list/country-cn/
#targets2=['http://www.proxynova.com/proxy-server-list/country-cn/']

#正则
retext = '''<span class="tbBottomLine" style="width:140px;">[\r\n\s]*(.+?)[\r\n\s]+</span>[\r\n\s]*'''
retext += '''<span class="tbBottomLine" style="width:50px;">[\r\n\s]*(.+?)[\r\n\s]*</span>[\r\n\s]*'''
retext += '''<span class="tbBottomLine " style="width:70px;">[\r\n\s]*.+[\r\n\s]*</span>[\r\n\s]*'''
retext += '''<span class="tbBottomLine " style="width:70px;">[\r\n\s]*(.+?)[\r\n\s]*</span>[\r\n\s]*'''
p = re.compile(retext,re.M)

#获取代理的类
class ProxyGet(threading.Thread):
    def __init__(self,target):
        threading.Thread.__init__(self)
        self.target = target
        
    def getProxy(self):
        print ("目标网站： " + self.target)
        req = urllib.request.urlopen(self.target)
        result = req.read().decode('utf-8')
        #print chardet.detect(result)
        matchs = p.findall(result)
        for row in matchs:
            ip = row[0]
            port = row[1]
            address = row[2]
            proxy = [ip,port,address]
            #print proxy
            rawProxyList.append(proxy)
            
    def run(self):
        self.getProxy()       
        
       
#检验代理的类    
class ProxyCheck(threading.Thread):
    def __init__(self,proxyList):
        threading.Thread.__init__(self)
        self.proxyList = proxyList
        self.timeout = 10
        self.testUrl = "http://www.baidu.com/"
        self.testStr = "030173"
        
    def checkProxy(self):
        cookies = urllib.request.HTTPCookieProcessor()#HTTPCookieProcessor类来处理的HTTP cookies。
        for proxy in self.proxyList:
            proxyHandler = urllib.request.ProxyHandler({"http" : r'http://%s:%s' %(proxy[0],proxy[1])})  
            #print r'http://%s:%s' %(proxy[0],proxy[1])
            opener = urllib.request.build_opener(cookies,proxyHandler)
            opener.addheaders = [('User-agent', 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:15.0) Gecko/20100101 Firefox/15.0.1')]
            #urllib2.install_opener(opener)
            t1 = time.time()
            
            try:
                #req = urllib2.urlopen("http://www.baidu.com", timeout=self.timeout)
                #req = opener.open(self.testUrl, timeout=self.timeout)
                #print "urlopen is ok...."
                #result = req.read()
                #print "read html...."
                socket.setdefaulttimeout(self.timeout)#访问超时设置
                req = urllib.request.Request(self.testUrl)
                result = urllib.request.urlopen(req).read().decode('utf-8')
                timeused = time.time() - t1
                pos = result.find(self.testStr)
                #print "pos is %s" %pos
                
                if (pos > -1):
                    checkedProxyList.append((proxy[0],proxy[1],proxy[2],timeused))
                    #print "ok ip: %s %s %s %s" %(proxy[0],proxy[1],proxy[2],timeused)
                else:
                    continue
                
            except Exception as e:
                print ("检测代理IP出错",e)
                continue
                       
    def sort(self):
        #sorted(checkedProxyList,cmp=lambda x,y:cmp(x[3],y[3]))
        sorted(checkedProxyList, key=lambda x:x[3])
        #sorted,排序函数。
                 
    def run(self):
        self.checkProxy()
        self.sort()
               
if __name__ == "__main__":
    start=time.clock()
    #"""
    getThreads = []
    checkThreads = []
    
    #对每个目标网站开启一个线程负责抓取代理
    for i in range(len(targets)):
        t = ProxyGet(targets[i])
        getThreads.append(t)
        
    for i in range(len(getThreads)):
        getThreads[i].start()
        
    for i in range(len(getThreads)):
        getThreads[i].join()        
        
    print (".......................总共抓取了%s个代理......................." %len(rawProxyList))

    
    
    
    #开启20个线程负责校验，将抓取到的代理分成20份，每个线程校验一份
    for i in range(20):
        t = ProxyCheck(rawProxyList[((len(rawProxyList)+19)//20) * i:((len(rawProxyList)+19)//20) * (i+1)])
        checkThreads.append(t)
    
    for i in range(len(checkThreads)):
        checkThreads[i].start()
        
    
    for i in range(len(checkThreads)):
        checkThreads[i].join()
        
     
    print (".......................总共有%s个代理通过校验......................." %len(checkedProxyList)  )  
        
    #持久化
    a=time.asctime()
    file_txt="F:/python/git/"+"测试通过的代理服务器ip "+a[-4:]+a[3:10]+".txt"
    f= open(file_txt,'w+')
    for proxy in checkedProxyList:
        print ("检验通过的代理是: %s:%s\t%s\t%s\n" %(proxy[0],proxy[1],proxy[2],proxy[3]))
        f.write("%s:%s\t%s\t%s\n"%(proxy[0],proxy[1],proxy[2],proxy[3]))
    f.close()
    
    #"""
    end=time.clock()
    print("程序运行时间为:",(end-start))

