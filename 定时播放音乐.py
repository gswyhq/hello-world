#/usr/lib/python3.5
# -*- coding: utf-8 -*-
import time  
import sys  
  
soundFile = '/home/gswyhq/有限状态机（FSM）/音乐.wav'  

  
def soundStart():  
    if sys.platform[:5] == 'linux':  
        from subprocess import Popen
        p2=Popen('aplay ' + soundFile,shell=True)  
        print('等待3秒')
        time.sleep(3)
        p2.kill
    else:  
        import winsound  
        winsound.PlaySound(soundFile, winsound.SND_FILENAME)  
          


def main():
    not_executed = 1 
    while(not_executed):  
        dt = list(time.localtime())  
        hour = dt[3]  
        minute = dt[4]  
        if hour == 21 and minute == 20: # 下午5点33分的时候开始提示  
            soundStart()  
            not_executed = 0  


if __name__ == "__main__":
    main()
    print('完毕')
    
    
'''
from threading import Timer 

In[51]: def hello(key=None):
...     print(key)
In[52]: t=Timer(2,hello,args=('你好',)) #两秒之后，执行函数hello，元组参数
In[53]: t.start()
你好
In[54]: t=Timer(2,hello,kwargs={'key':'sdfwe fwehi'}) #关键字参数
In[55]: t.start()
sdfwe fwehi



'''
