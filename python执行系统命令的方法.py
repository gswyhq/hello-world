#!/usr/bin/python
# -*- coding:utf8 -*- #
from __future__ import  generators
from __future__ import  division
from __future__ import  print_function
from __future__ import  unicode_literals
import sys,os,json

if sys.version >= '3':
    PY3 = True
else:
    PY3 = False

if PY3:
    import pickle
else:
    import cPickle as pickle
    from codecs import open

import commands    

import os

def main():
    cmd='ls'
    d=commands.getoutput(cmd) # 此命令在python3 中不能使用
    
    # 仅仅在一个子终端运行系统命令，而不能获取命令执行后的返回信息，也就是说不能“t=system('pwd')”这样使用，只能下面这种使用
    system(pwd) # pwd 可以不加引号
    
    # 如果再命令行下执行，结果直接打印出来
    os.system('pwd') #pwd 必须加引号
    #下面这样使用，虽说可以，但是t捕获到的仅仅是执行后的返回码，0代表成功，否则是其他的返回值，是一个int类型数值
    t=os.system('pwd')
    
    # 该方法不但执行命令还返回执行后的信息对象
    #好处在于：将返回的结果赋于一变量，便于程序的处理。
    tmp = os.popen('ls *.py').readlines()
    #注意： 当执行命令的参数或者返回中包含了中文文字，那么建议使用subprocess，如果使用os.popen则会出现编码错误
    
    
    
#1、当你对shell命令的输出不感兴趣，只希望程序被运行，你可以典型的使用subprocess.call
#2、如果你需要捕获命令的输出结果，那么你就需要使用subprocess.Popen

#注意，以下才是重点：
#在subprocess.call与Popen之间，存在一个非常大的区别。
#subprocess.call会封锁对响应的等待，而subprocess.Popen则不会！！
 
#如果程序\子进程没有响应，python不理它，继续执行python语句，而Popen会等待，知道子进程输出结果才执行下一步语句。   
    
    
    import subprocess
    subprocess.call (["echo", "arg1", "arg2"],shell=True)
    #获取返回和输出:
    #利用subprocess.PIPE将多个子进程的输入和输出连接在一起，构成管道(pipe)
    p = subprocess.Popen('ls', shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for line in p.stdout.readlines():
        print (line.decode('utf8'))
    retval = p.wait()
    
    if PY3:
        subprocess.getoutput('date')  #date 是shell是一个命令
        #Out[4]: '2016年 05月 24日 星期二 09:12:06 CST'

        subprocess.getstatusoutput('date')
        #Out[5]: (0, '2016年 05月 24日 星期二 09:12:22 CST')

    else:
        #四、使用模块commands
        import commands
        #只返回执行的结果, 忽略返回值.
        commands.getoutput("date") 
        
        #用os.popen()执行命令cmd, 然后返回两个元素的元组(status, result). cmd执行的方式是{ cmd ; } 2>&1, 这样返回结果里面就会包含标准输出和标准错误.
        commands.getstatusoutput("date")
        
        #返回ls -ld file执行的结果.参数必须是一个文件
        commands.getstatus(file)




if __name__ == "__main__":
    main()
    
    
    
'''
>>> import subprocess
>>> child1 = subprocess.Popen(["ls","-l"], stdout=subprocess.PIPE)
>>> print child1.stdout.read(),
#或者child1.communicate()
>>> import subprocess
>>> child1 = subprocess.Popen(["cat","/etc/passwd"], stdout=subprocess.PIPE)
>>> child2 = subprocess.Popen(["grep","0:0"],stdin=child1.stdout, stdout=subprocess.PIPE)
>>> out = child2.communicate()

subprocess.PIPE实际上为文本流提供一个缓存区。child1的stdout将文本输出到缓存区，随后child2的stdin从该PIPE中将文本读取走。child2的输出文本也被存放在PIPE中，直到communicate()方法从PIPE中读取出PIPE中的文本。
注意：communicate()是Popen对象的一个方法，该方法会阻塞父进程，直到子进程完成



subprocess的目的就是启动一个新的进程并且与之通信。

subprocess模块中只定义了一个类: Popen。可以使用Popen来创建进程，并与进程进行复杂的交互。它的构造函数如下：

subprocess.Popen(args, bufsize=0, executable=None, stdin=None, stdout=None, stderr=None, 
                preexec_fn=None, close_fds=False, shell=False, cwd=None, env=None, 
                universal_newlines=False, startupinfo=None, creationflags=0)

参数args可以是字符串或者序列类型（如：list，元组），用于指定进程的可执行文件及其参数。如果是序列类型，第一个元素通常是可执行文件的路径。我们也可以显式的使用executeable参数来指定可执行文件的路径。

参数stdin, stdout, stderr分别表示程序的标准输入、输出、错误句柄。他们可以是PIPE，文件描述符或文件对象，也可以设置为None，表示从父进程继承。

如果参数shell设为true，程序将通过shell来执行。

参数env是字典类型，用于指定子进程的环境变量。如果env = None，子进程的环境变量将从父进程中继承。

subprocess.PIPE

　　在创建Popen对象时，subprocess.PIPE可以初始化stdin, stdout或stderr参数。表示与子进程通信的标准流。

subprocess.STDOUT

　　创建Popen对象时，用于初始化stderr参数，表示将错误通过标准输出流输出。

Popen的方法：

Popen.poll()

　　用于检查子进程是否已经结束。设置并返回returncode属性。

Popen.wait()

　　等待子进程结束。设置并返回returncode属性。

Popen.communicate(input=None)

　　与子进程进行交互。向stdin发送数据，或从stdout和stderr中读取数据。可选参数input指定发送到子进程的参数。Communicate()返回一个元组：(stdoutdata, stderrdata)。注意：如果希望通过进程的stdin向其发送数据，在创建Popen对象的时候，参数stdin必须被设置为PIPE。同样，如果希望从stdout和stderr获取数据，必须将stdout和stderr设置为PIPE。

Popen.send_signal(signal)

　　向子进程发送信号。

Popen.terminate()

　　停止(stop)子进程。在windows平台下，该方法将调用Windows API TerminateProcess（）来结束子进程。

Popen.kill()

　　杀死子进程。

Popen.stdin，Popen.stdout ，Popen.stderr ，官方文档上这么说：

stdin, stdout and stderr specify the executed programs’ standard input, standard output and standard error file handles, respectively. Valid values are PIPE, an existing file descriptor (a positive integer), an existing file object, and None.

Popen.pid

　　获取子进程的进程ID。

Popen.returncode

　　获取进程的返回值。如果进程还没有结束，返回None。

---------------------------------------------------------------

简单的用法：

[python] view plain copy
p=subprocess.Popen("dir", shell=True)  
p.wait()  
shell参数根据你要执行的命令的情况来决定，上面是dir命令，就一定要shell=True了，p.wait()可以得到命令的返回值。

如果上面写成a=p.wait()，a就是returncode。那么输出a的话，有可能就是0【表示执行成功】。

---------------------------------------------------------------------------

进程通讯

如果想得到进程的输出，管道是个很方便的方法，这样：

[python] view plain copy
p=subprocess.Popen("dir", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)  
(stdoutput,erroutput) = p.<span>commu</span>nicate()  
p.communicate会一直等到进程退出，并将标准输出和标准错误输出返回，这样就可以得到子进程的输出了。

再看一个communicate的例子。

p=subprocess.Popen('ls',shell=True,stdout=subprocess.PIPE)
stdoutput,erroutput=p.communicate('/home/gswewf')

print(stdoutput)
print(erroutput)

上面的例子通过communicate给stdin发送数据，然后使用一个tuple接收命令的执行结果。

------------------------------------------------------------------------

上面，标准输出和标准错误输出是分开的，也可以合并起来，只需要将stderr参数设置为subprocess.STDOUT就可以了，这样子：

[python] view plain copy
p=subprocess.Popen("dir", shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)  
(stdoutput,erroutput) = p.<span>commu</span>nicate()  
如果你想一行行处理子进程的输出，也没有问题：

[python] view plain copy
p=subprocess.Popen("dir", shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)  
while True:  
    buff = p.stdout.readline()  
    if buff == '' and p.poll() != None:  
        break  
------------------------------------------------------

死锁

但是如果你使用了管道，而又不去处理管道的输出，那么小心点，如果子进程输出数据过多，死锁就会发生了，比如下面的用法：

[python] view plain copy
p=subprocess.Popen("longprint", shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)  
p.wait()  
longprint是一个假想的有大量输出的进程，那么在我的xp, Python2.5的环境下，当输出达到4096时，死锁就发生了。当然，如果我们用p.stdout.readline或者p.communicate去清理输出，那么无论输出多少，死锁都是不会发生的。或者我们不使用管道，比如不做重定向，或者重定向到文件，也都是可以避免死锁的。

----------------------------------

subprocess还可以连接起来多个命令来执行。

在shell中我们知道，想要连接多个命令可以使用管道。

在subprocess中，可以使用上一个命令执行的输出结果作为下一次执行的输入。例子如下：
p1=subprocess.Popen('cat /home/gswewf/gow69/参数传递.py',shell=True,stdout=subprocess.PIPE)
p2=subprocess.Popen('tail -2',shell=True,stdin=p1.stdout,stdout=subprocess.PIPE)

print(p2.stdout)
print(p2.stdout.read())


例子中，p2使用了第一次执行命令的结果p1的stdout作为输入数据，然后执行tail命令。


'''
