#! /usr/lib/python3.5
# -*- coding: utf-8 -*-
#http://www.cnblogs.com/kaituorensheng/p/4445418.html

import multiprocessing
import time


"""创建进程的类：Process([group [, target [, name [, args [, kwargs]]]]])，target表示调用对象，args表示调用对象的位置参数元组。kwargs表示调用对象的字典。name为别名。group实质上不使用。
方法：is_alive()、join([timeout])、run()、start()、terminate()。其中，Process以start()启动某个进程。

属性：authkey、daemon（要通过start()设置）、exitcode(进程在运行时为None、如果为–N，表示被信号N结束）、name、pid。其中daemon是父进程终止后自动终止，且自己不能产生新进程，必须在start()之前设置。


例1.1：创建函数并将其作为单个进程


def worker(interval):
    n = 5
    while n > 0:
        print("The time is {0}".format(time.ctime()))
        time.sleep(interval)
        n -= 1

def main():
    p = multiprocessing.Process(target = worker, args = (3,))
    #p.daemon = True #若设置daemon为真，则父进程终止后，即使子进程没有完成，也被强制终止。
    p.start()
    print ("p.pid:", p.pid)#进程代码
    print ("p.name:", p.name)#进程别名
    print ("p.is_alive:", p.is_alive())#
if __name__=='__main__':
    main()


p.pid: 8736
p.name: Process-1
p.is_alive: True
The time is Tue Apr 21 20:55:12 2015
The time is Tue Apr 21 20:55:15 2015
The time is Tue Apr 21 20:55:18 2015
The time is Tue Apr 21 20:55:21 2015
The time is Tue Apr 21 20:55:24 2015
 

例1.2：创建函数并将其作为多个进程

复制代码
import multiprocessing
import time

def worker_1(interval,k):
    print ("worker_1",
           '等待时间是：',k)
    time.sleep(interval)
    print ("end worker_1",k)

def worker_2(interval):
    print ("worker_2")
    time.sleep(interval)
    print ("end worker_2")

def worker_3(interval):
    print ("worker_3")
    time.sleep(interval)
    print ("end worker_3")

if __name__ == "__main__":
    p1 = multiprocessing.Process(target = worker_1, args = (2,'浓厚'))
    p2 = multiprocessing.Process(target = worker_2, args = (3,))
    p3 = multiprocessing.Process(target = worker_3, args = (4,))

    p1.start()
    p2.start()
    p3.start()

    print("The number of CPU is:" + str(multiprocessing.cpu_count()))
    for p in multiprocessing.active_children():
        print("child   p.name:" + p.name + "\tp.id" + str(p.pid))
    print ("END!!!!!!!!!!!!!!!!!")


The number of CPU is:4
child   p.name:Process-3    p.id7992
child   p.name:Process-2    p.id4204
child   p.name:Process-1    p.id6380
END!!!!!!!!!!!!!!!!!
worker_1
worker_3
worker_2
end worker_1
end worker_2
end worker_3
 

例1.3：将进程定义为类

复制代码
import multiprocessing
import time

class ClockProcess(multiprocessing.Process):
    def __init__(self, interval):
        multiprocessing.Process.__init__(self)
        print('传人的参数是：',interval)
        self.interval = interval

    def run(self):
        n = 5
        while n > 0:
            print("the time is {0}".format(time.ctime()))
            time.sleep(self.interval)
            n -= 1

if __name__ == '__main__':
    p = ClockProcess(3)
    p.start()      

#注：进程p调用start()时，自动调用run()


the time is Tue Apr 21 20:31:30 2015
the time is Tue Apr 21 20:31:33 2015
the time is Tue Apr 21 20:31:36 2015
the time is Tue Apr 21 20:31:39 2015
the time is Tue Apr 21 20:31:42 2015
 

例1.4：daemon程序对比结果

#1.4-1 不加daemon属性

复制代码
import multiprocessing
import time

def worker(interval):
    print("work start:{0}".format(time.ctime()));
    print('等待时间：',interval)
    time.sleep(interval)
    print("work end:{0}".format(time.ctime()));

if __name__ == "__main__":
    p = multiprocessing.Process(target = worker, args = (3,))
    p.start()
    print ("end!")



end!
work start:Tue Apr 21 21:29:10 2015
work end:Tue Apr 21 21:29:13 2015
#1.4-2 加上daemon属性

复制代码
import multiprocessing
import time

def worker(interval):
    print("work start:{0}".format(time.ctime()));
    print('等待时间：',interval)
    time.sleep(interval)
    print("work end:{0}".format(time.ctime()));

if __name__ == "__main__":
    p = multiprocessing.Process(target = worker, args = (3,))
    p.daemon = True
    p.start()
    print ("end!")


复制代码
结果

1
end!
注：因子进程设置了daemon属性，主进程结束，它们就随着结束了。

#1.4-3 设置daemon执行完结束的方法

复制代码
import multiprocessing
import time

def worker(interval):
    print("work start:{0}".format(time.ctime()));
    time.sleep(interval)
    print("work end:{0}".format(time.ctime()));

if __name__ == "__main__":
    p = multiprocessing.Process(target = worker, args = (3,))
    p.daemon = True
    p.start()
    p.join()#join，阻塞父进程，即父进程要等子进程完成了才能继续。恰好抵消了设置daemon为真的作用。
    print ("end!")


work start:Tue Apr 21 22:16:32 2015
work end:Tue Apr 21 22:16:35 2015
end!
 


2. Lock
当多个进程需要访问共享资源的时候，Lock可以用来避免访问的冲突。

复制代码
import multiprocessing
import sys


def worker_with(lock, f):
    with lock:
        fs = open(f, 'a+')
        n = 10
        while n > 1:
            fs.write("Lockd acquired via with\n")
            n -= 1
        fs.close()
        
def worker_no_with(lock, f):
    lock.acquire()
    try:
        fs = open(f, 'a+')
        n = 10
        while n > 1:
            fs.write("Lock acquired directly\n")
            n -= 1
        fs.close()
    finally:
        lock.release()
    
if __name__ == "__main__":
    lock = multiprocessing.Lock()
    f = "/home/gswyhq/gow69/多进程测试file.txt"
    w = multiprocessing.Process(target = worker_with, args=(lock, f))
    nw = multiprocessing.Process(target = worker_no_with, args=(lock, f))
    w.start()
    nw.start()
    print ("end")

Lockd acquired via with
Lockd acquired via with
Lockd acquired via with
Lockd acquired via with
Lockd acquired via with
Lockd acquired via with
Lockd acquired via with
Lockd acquired via with
Lockd acquired via with
Lock acquired directly
Lock acquired directly
Lock acquired directly
Lock acquired directly
Lock acquired directly
Lock acquired directly
Lock acquired directly
Lock acquired directly
Lock acquired directly
 

回到顶部
3. Semaphore
Semaphore用来控制对共享资源的访问数量，例如池的最大连接数。

复制代码
import multiprocessing
import time


def worker(s, i):
    s.acquire()
    print(multiprocessing.current_process().name + "acquire");
    time.sleep(i)
    print(multiprocessing.current_process().name + "release\n");
    s.release()

if __name__ == "__main__":
    s = multiprocessing.Semaphore(2)
    for i in range(5):
        p = multiprocessing.Process(target = worker, args=(s, 2))
        print('开始进程：',i)
        p.start()





Process-1acquire
Process-1release
 
Process-2acquire
Process-3acquire
Process-2release
 
Process-5acquire
Process-3release
 
Process-4acquire
Process-5release
 
Process-4release
 

回到顶部
4. Event
Event用来实现进程间同步通信。

复制代码
import multiprocessing
import time

def wait_for_event(e):
    print("wait_for_event: starting",time.ctime())
    e.wait()
    print("wairt_for_event: e.is_set()->" + str(e.is_set()),time.ctime())

def wait_for_event_timeout(e, t):
    print("wait_for_event_timeout:starting",time.ctime())
    e.wait(t)
    print("wait_for_event_timeout:e.is_set->" + str(e.is_set()),time.ctime())#若子进程在父进程中e.set()语句之前就结束了，则e.is_set()为假，否则为真

if __name__ == "__main__":
    e = multiprocessing.Event()
    w1 = multiprocessing.Process(name = "block",
            target = wait_for_event,
            args = (e,))

    w2 = multiprocessing.Process(name = "non-block",
            target = wait_for_event_timeout,
            args = (e,2))#设置等待时间,当e.is_set()为真时，则此设置无效。
    w1.start()
    w2.start()

    time.sleep(3)

    e.set()
    #time.sleep(3)
    print("main: event is set")




wait_for_event: starting
wait_for_event_timeout:starting
wait_for_event_timeout:e.is_set->False
main: event is set
wairt_for_event: e.is_set()->True
 

回到顶部
5. Queue
Queue是多进程安全的队列，可以使用Queue实现多进程之间的数据传递。put方法用以插入数据到队列中，put方法还有两个可选参数：blocked和timeout。
如果blocked为True（默认值），并且timeout为正值，该方法会阻塞timeout指定的时间，直到该队列有剩余的空间。如果超时，会抛出Queue.Full异常。
如果blocked为False，但该Queue已满，会立即抛出Queue.Full异常。
 
get方法可以从队列读取并且删除一个元素。同样，get方法有两个可选参数：blocked和timeout。如果blocked为True（默认值），
并且timeout为正值，那么在等待时间内没有取到任何元素，会抛出Queue.Empty异常。如果blocked为False，有两种情况存在，
如果Queue有一个值可用，则立即返回该值，否则，如果队列为空，则立即抛出Queue.Empty异常。Queue的一段示例代码：


import multiprocessing

def writer_proc(q):      
    try:         
        q.put(1, block = False) 
    except:         
        pass   

def reader_proc(q):      
    try:         
        print (q.get(block = False) )
    except:         
        pass

if __name__ == "__main__":
    q = multiprocessing.Queue()
    writer = multiprocessing.Process(target=writer_proc, args=(q,))  
    writer.start()   

    reader = multiprocessing.Process(target=reader_proc, args=(q,))  
    reader.start()  

    reader.join()  
    writer.join()




回到顶部
6. Pipe
Pipe方法返回(conn1, conn2)代表一个管道的两个端。Pipe方法有duplex参数，如果duplex参数为True(默认值)，那么这个管道是全双工模式，也就是说conn1和conn2均可收发。duplex为False，conn1只负责接受消息，conn2只负责发送消息。
 
send和recv方法分别是发送和接受消息的方法。例如，在全双工模式下，可以调用conn1.send发送消息，conn1.recv接收消息。如果没有消息可接收，recv方法会一直阻塞。如果管道已经被关闭，那么recv方法会抛出EOFError。
复制代码
import multiprocessing
import time



def proc1(pipe):
    while True:
        for i in range(1000):
            print ("send: %s" %(i))
            pipe.send(i)
            time.sleep(1)

def proc2(pipe):
    while True:
        print ("proc2 rev:", pipe.recv())
        time.sleep(1)

def proc3(pipe):
    while True:
        print ("PROC3 rev:", pipe.recv())
        time.sleep(1)

if __name__ == "__main__":
    pipe = multiprocessing.Pipe()
    p1 = multiprocessing.Process(target=proc1, args=(pipe[0],))
    p2 = multiprocessing.Process(target=proc2, args=(pipe[1],))
    #p3 = multiprocessing.Process(target=proc3, args=(pipe[1],))

    p1.start()
    p2.start()
    #p3.start()

    p1.join()
    p2.join()
    #p3.join()



回到顶部
7. Pool
在利用Python进行系统管理的时候，特别是同时操作多个文件目录，或者远程控制多台主机，并行操作可以节约大量的时间。
当被操作对象数目不大时，可以直接利用multiprocessing中的Process动态成生多个进程，十几个还好，但如果是上百个，
上千个目标，手动的去限制进程数量却又太过繁琐，此时可以发挥进程池的功效。
Pool可以提供指定数量的进程，供用户调用，当有新的请求提交到pool中时，如果池还没有满，那么就会创建一个新的进程用来执行该请求；
但如果池中的进程数已经达到规定最大值，那么该请求就会等待，直到池中有进程结束，才会创建新的进程来它。

 

例7.1：使用进程池（非阻塞）

复制代码
#coding: utf-8
import multiprocessing
import time

def func(msg):
    print ("msg:", msg)
    time.sleep(3)
    print ("end")

if __name__ == "__main__":
    pool = multiprocessing.Pool(processes = 3)
    for i in range(4):
        msg = "hello %d" %(i)
        pool.apply_async(func, (msg, ))
        #维持执行的进程总数为processes，当一个进程执行完毕后会添加新的进程进去
        #apply_async(func[, args[, kwds[, callback]]]) 它是非阻塞，apply(func[, args[, kwds]])是阻塞的（理解区别，看例1例2结果区别）

    print( "Mark~ Mark~ Mark~~~~~~~~~~~~~~~~~~~~~~")
    pool.close()
    pool.join()   #调用join之前，先调用close函数，否则会出错。执行完close后不会有新的进程加入到pool,join函数等待所有子进程结束
    print ("Sub-process(es) done.")



mMsg: hark~ Mark~ Mark~~~~~~~~~~~~~~~~~~~~~~ello 0
 
msg: hello 1
msg: hello 2
end
msg: hello 3
end
end
end
Sub-process(es) done.
函数解释：

apply_async(func[, args[, kwds[, callback]]]) 它是非阻塞，apply(func[, args[, kwds]])是阻塞的（理解区别，看例1例2结果区别）
close()    关闭pool，使其不在接受新的任务。
terminate()    结束工作进程，不在处理未完成的任务。
join()    主进程阻塞，等待子进程的退出， join方法要在close或terminate之后使用。
执行说明：创建一个进程池pool，并设定进程的数量为3，xrange(4)会相继产生四个对象[0, 1, 2, 4]，四个对象被提交到pool中，因pool指定进程数为3，所以0、1、2会直接送到进程中执行，当其中一个执行完事后才空出一个进程处理对象3，所以会出现输出“msg: hello 3”出现在"end"后。因为为非阻塞，主函数会自己执行自个的，不搭理进程的执行，所以运行完for循环后直接输出“mMsg: hark~ Mark~ Mark~~~~~~~~~~~~~~~~~~~~~~”，主程序在pool.join（）处等待各个进程的结束。



例7.2：使用进程池（阻塞）

复制代码
#coding: utf-8
import multiprocessing
import time

##################################################################################################################################
"""

def func(msg):
    print ("msg:", msg,time.ctime())
    time.sleep(3)
    print ("end",time.ctime())

if __name__ == "__main__":
    pool = multiprocessing.Pool(processes = 3)
    for i in range(4):
        msg = "hello %d" %(i)
        pool.apply(func, (msg, ))   #维持执行的进程总数为processes，当一个进程执行完毕后会添加新的进程进去

    print ("Mark~ Mark~ Mark~~~~~~~~~~~~~~~~~~~~~~",time.ctime())#因为是阻塞模式，此句要等所有子进程运行完毕才会执行
    pool.close()
    pool.join()   #调用join之前，先调用close函数，否则会出错。执行完close后不会有新的进程加入到pool,join函数等待所有子进程结束
    print ("Sub-process(es) done.",time.ctime())


##################################################################################################################################
"""


msg: hello 0
end
msg: hello 1
end
msg: hello 2
end
msg: hello 3
end
Mark~ Mark~ Mark~~~~~~~~~~~~~~~~~~~~~~
Sub-process(es) done.
　　

例7.3：使用进程池，并关注结果

复制代码
import multiprocessing
import time

def func(msg):
    print "msg:", msg
    time.sleep(3)
    print "end"
    return "done" + msg

if __name__ == "__main__":
    pool = multiprocessing.Pool(processes=4)
    result = []
    for i in xrange(3):
        msg = "hello %d" %(i)
        result.append(pool.apply_async(func, (msg, )))
    pool.close()
    pool.join()
    for res in result:
        print ":::", res.get()
    print "Sub-process(es) done."
复制代码
一次执行结果

1
2
3
4
5
6
7
8
9
10
msg: hello 0
msg: hello 1
msg: hello 2
end
end
end
::: donehello 0
::: donehello 1
::: donehello 2
Sub-process(es) done.
 

例7.4：使用多个进程池

复制代码
#coding: utf-8
import multiprocessing
import os, time, random

def Lee():
    print "\nRun task Lee-%s" %(os.getpid()) #os.getpid()获取当前的进程的ID
    start = time.time()
    time.sleep(random.random() * 10) #random.random()随机生成0-1之间的小数
    end = time.time()
    print 'Task Lee, runs %0.2f seconds.' %(end - start)

def Marlon():
    print "\nRun task Marlon-%s" %(os.getpid())
    start = time.time()
    time.sleep(random.random() * 40)
    end=time.time()
    print 'Task Marlon runs %0.2f seconds.' %(end - start)

def Allen():
    print "\nRun task Allen-%s" %(os.getpid())
    start = time.time()
    time.sleep(random.random() * 30)
    end = time.time()
    print 'Task Allen runs %0.2f seconds.' %(end - start)

def Frank():
    print "\nRun task Frank-%s" %(os.getpid())
    start = time.time()
    time.sleep(random.random() * 20)
    end = time.time()
    print 'Task Frank runs %0.2f seconds.' %(end - start)
        
if __name__=='__main__':
    function_list=  [Lee, Marlon, Allen, Frank] 
    print "parent process %s" %(os.getpid())

    pool=multiprocessing.Pool(4)
    for func in function_list:
        pool.apply_async(func)     #Pool执行函数，apply执行函数,当有一个进程执行完毕后，会添加一个新的进程到pool中

    print 'Waiting for all subprocesses done...'
    pool.close()
    pool.join()    #调用join之前，一定要先调用close() 函数，否则会出错, close()执行后不会有新的进程加入到pool,join函数等待素有子进程结束
    print 'All subprocesses done.'
复制代码
一次执行结果

1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
parent process 7704
 
Waiting for all subprocesses done...
Run task Lee-6948
 
Run task Marlon-2896
 
Run task Allen-7304
 
Run task Frank-3052
Task Lee, runs 1.59 seconds.
Task Marlon runs 8.48 seconds.
Task Frank runs 15.68 seconds.
Task Allen runs 18.08 seconds.
All subprocesses done.


"""
