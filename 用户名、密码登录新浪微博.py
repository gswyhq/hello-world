# -*- coding: utf-8 -*-
#登录新浪微博
#来源：http://www.lpfrx.com/archives/4396/

#http://www.open-open.com/doc/view/fd5e3282fd1b4d1aaa2bf7dddf47822e
"""
#coding=utf8
import sys,os
import time
import string
#from PyQt4.QtCore import *
#from PyQt4.QtGui import *
#from PyQt4.QtWebKit import *
from PyQt5 import QtGui, QtCore, QtWebKit, QtNetwork,QtWebKitWidgets
#from PyQt4.QtCore import QObject, QVariant,QPoint
from PyQt5.QtCore import *

class Sp():
    def __init__(self):
        self.user_agent = 'Mozilla/5.0 (X11; U; Linux i686; en-US; rv:1.9.2) Gecko/20100124 Firefox/3.6 (Swiftfox)'
        #self.app = app
        #self._url = url
        self.webView =QtWebKitWidgets.QWebView()
        self.textx =""
        self.mm=0
        self.p=0
        self.js_click = ""
             var evt = document.createEvent("MouseEvents");
             evt.initMouseEvent("click", true, true, window, 1, 1, 1, 1, 1, false, false, false, false, 0, this);
             this.dispatchEvent(evt);
             ""
        print('url')
        
    def save(self):
        print ("start...",self.mm)
        data = self.webView.page().currentFrame().documentElement().toInnerXml()
        self.textx = data
        print('save')
        print(self.textx)

        while True:
            if len(self.textx)>0:
                print ("wang cheng")
                break
        #open("f:/python/git/htm0.txt","w").write(data)
        print ('start...')

        if self.p==1:
            userx = "xxx@sina.com"
            pwdx  = "123456789"
            jss = "document.getElementsByName('username')[0].value='"+ userx +"';document.getElementsByName('password')[0].value='"+ pwdx +"';document.getElementsByName('password')[0].focus();"
            print (jss)
            self.webView.page().mainFrame().evaluateJavaScript(jss)
            elem = self.webView.page().mainFrame().findAllElements("a[tabindex='6']")

            if len(elem)==1:
                print ('elem len:',len(elem))
	     
	     #elem[0].evaluateJavaScript(self.js_click)
	     #elem[0].evaluateJavaScript("this.value=123456789;")
        sa= unicode(data).encode("utf8")
        #open("f:/python/git/htm1.txt","w").write(sa)
        self.p=7
        self.mm=0

        if self.p==7:
            self.timeoutTimer.stop()

        self.mm=self.mm+1
        print ("finish......end",self.p,self.mm)
	#self.app.processEvents()
	#self.app.quit()
	#sys.exit()
	

    def timetest(self):
        #timetest函数是打开网址延时4秒, 等所有东西加载完毕。
        self.timeoutTimer = QTimer()
        self.timeoutTimer.start(4000)
        self.timeoutTimer.timeout.connect(self.save)
        #self.timeoutTimer.stop()
       
    def loadx(self):
        self.finishedx =False
        #self.webView.loadFinished.connect()
        self.webView.load(QtCore.QUrl(self._url))

    def crawl(self,url):
        #QtCore.QObject.connect(self.webView,QtCore.SIGNAL("loadFinished(bool)"),self.timetest())
        self.webView.loadFinished.connect(self.timetest)
        #self.webView.loadProgress.connect(self.print_percent)
        self.webView.load(QtCore.QUrl(url))
        print('self._',url)
        self.webView.show()

#sys.exit()  


#app = QtGui.QApplication(sys.argv)
if __name__=="__main__":
    url="http://tieba.baidu.com/p/3246566913"
    s = Sp()
    print('sdfs',url)
    s.crawl(url)
    time.sleep(0.01)
    print ("ok ok 0"  )
    sys.exit(app.exec_())




class Sp():
    def __init__(self, url,app):
        self.user_agent = 'Mozilla/5.0 (X11; U; Linux i686; en-US; rv:1.9.2) Gecko/20100124 Firefox/3.6 (Swiftfox)'
        self.app = app
        self._url = url
	self.webView = QtWebKit.QWebView()
	self.textx =""
        self.mm=0
	self.p=1       
        self.js_click = ""
             var evt = document.createEvent("MouseEvents");
             evt.initMouseEvent("click", true, true, window, 1, 1, 1, 1, 1, false, false, false, false, 0, this);
             this.dispatchEvent(evt);
             ""
        
    def save(self):
        print ("start...",self.mm)
        data = self.webView.page().currentFrame().documentElement().toInnerXml()
	self.textx = data
	while True:
            if len(self.textx)>0:
                print ("wang cheng")
                break
        open("f:/python/git/htm0.txt","w").write(data)
        print ('start...')

        if self.p==1:
            userx = "xxx@sina.com"
            pwdx  = "123456789"
            jss = "document.getElementsByName('username')[0].value='"+ userx +"';document.getElementsByName('password')[0].value='"+ pwdx +"';document.getElementsByName('password')[0].focus();"
            print (jss)
            self.webView.page().mainFrame().evaluateJavaScript(jss)
            elem = self.webView.page().mainFrame().findAllElements("a[tabindex='6']")

            if len(elem)==1:
                print ('elem len:',len(elem))
	     
	     #elem[0].evaluateJavaScript(self.js_click)
	     #elem[0].evaluateJavaScript("this.value=123456789;")
        sa= unicode(data).encode("utf8")
        open("f:/python/git/htm1.txt","w").write(sa)
        self.p=7
        self.mm=0

        if self.p==7:
            self.timeoutTimer.stop() 

	self.mm=self.mm+1
	print ("finish......end",self.p,self.mm)
	#self.app.processEvents()
	#self.app.quit()
	#sys.exit()

    def timetest(self):
       self.timeoutTimer = QTimer() 
       self.timeoutTimer.start(4000)
       self.timeoutTimer.timeout.connect(self.save)
       #self.timeoutTimer.stop()
       
    def loadx(self):
        self.finishedx =False
        #self.webView.loadFinished.connect()
        self.webView.load(QtCore.QUrl(self._url))

    def crawl(self):
        #QtCore.QObject.connect(self.webView,QtCore.SIGNAL("loadFinished(bool)"),self.timetest())
	self.webView.loadFinished.connect(self.timetest)
	#self.webView.loadProgress.connect(self.print_percent)
	self.webView.load(QtCore.QUrl(self._url))	
	self.webView.show()

#sys.exit()  


app = QtGui.QApplication(sys.argv)
s = Sp("http://weibo.com/",app)
s.crawl()

#while not s.finishedx:
#   app.processEvents()
#   time.sleep(0.01)

print ("ok ok 0"  )


sys.exit(app.exec_())

"""
