
一般步骤
1、编写程序
2、定义Dockerfile，方便迁移到任何地方；
3、编写docker-compose.yml文件；
4、运行docker-compose up启动服务

# 第一步：编写py程序
gswewf@gswewf-pc:~/docker/test_web$ vim app.py 

from flask import Flask
from redis import Redis
import os
app = Flask(__name__)
redis = Redis(host='redis', port=6379)

@app.route('/')
def hello():
    redis.incr('hits')
    return 'Hello World! I have been seen %s times.' % redis.get('hits')

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)

# 第二步：定义这个Python应用容器的依赖文件requirements.txt
gswewf@gswewf-pc:~/docker/test_web$ vim requirements.txt 
flask
redis

# 第三步：编写Dockerfile文件，即创建python应用的docker镜像
gswewf@gswewf-pc:~/docker/test_web$ vim Dockerfile
FROM python:3.5
ADD . /code
WORKDIR /code
RUN pip3 install -r requirements.txt

# 上述，指定镜像是python:3.5，并复制当前目录到容器的/code目录；然后下载需要的扩展（申明在requirements.txt）。

# 第四步：使用 docker-compose.yml定义容器应用的服务配置
gswewf@gswewf-pc:~/docker/test_web$ vim docker-compose.yml 
web:
  build: .
  command: python3 app.py
  ports:
   - "5000:5000"
  volumes:
   - .:/code
  links:
   - redis
redis:
  image: microbox/redis

# 该文件指定镜像来源自build后的镜像，挂载数据卷/code，端口是5000:5000，使用redis服务，最后执行入口命令python3 app.py。
# 这里为了下载更快，使用了精简版的redis镜像（7M）microbox/redis，
#想用官方redis镜像的话也可以使用redis（’image: microbox/redis‘ 改为：’image: redis‘即可）。

# 上述的 docker-compose.yml文件中定义了两种服务：
web：从当前目录的dockerfile文件进行构建，并且将其作为Volume挂载到容器的/code目录中，
然后通过python app.py来启动Flask应用。最后将容器的5000端口暴露出来，并将其映射到主机的5000端口上。
其中，使用links来定义容器之间的依赖关系，表示web容器依赖于Redis容器。
redis：直接使用Docker Hub上的官方镜像来提供所需的Redis服务支持，会直接从docker hub上进行下载。
更多配置可参考：http://www.cnblogs.com/freefei/p/5311294.html


# 显示当前的目录文件结构
gswewf@gswewf-pc:~/docker/test_web$ ls
app.py  docker-compose.yml  Dockerfile  requirements.txt

# 第五步：执行compose启动命令 ，启动命令如下
gswewf@gswewf-pc:~/docker/test_web$ docker-compose up

# 通过compose启动成功后，会最先开始构建Redis容器，紧随其后则构建Python容器，并生成应用的镜像；
# 然后，Docker Compose会并行地启动全部容器，应用容器会通过compose与被依赖容器进行通信，实现该应用的部署。
运行中会下载镜像python:3.5和扩展Flask、Redis
访问http://0.0.0.0:5000/即可看到效果：
  Hello World! I have been seen 1 times.


# 启动时，有时会报错：
Couldn’t connect to Docker daemon at http+unix://var/run/docker.sock - is it running? 

解决方法如下：
1、设置 DOCKER_HOST 环境变量
gswewf@gswewf-pc:~$ vim .bashrc
添加：export DOCKER_HOST=tcp://localhost:4243

2、使更改配置生效
gswewf@gswewf-pc:~$ source ~/.bashrc

3、重启电脑

4、编辑/etc/default/docker,修改下面的参数为
DOCKER_OPTS="-H tcp://127.0.0.1:4243 -H unix:///var/run/docker.sock"

5、重启 docker 服务
$ sudo service docker restart

6、检查并确定 Docker 运行在 localhost:4243
$ netstat -ant | grep 4243

7、继续启动服务
gswewf@gswewf-pc:~/docker/test_web$ docker-compose up

# 有时会报下面的错误：
gswewf@gswewf-pc:~$ docker ps
Cannot connect to the Docker daemon at tcp://localhost:4243. Is the docker daemon running?
这时需要把当前用户加入到docker用户组中
gswewf@gswewf-pc:~$ sudo usermod -a -G docker $USER
# 查看docker服务是否开启：
gswewf@gswewf-pc:~$ netstat -pan | grep 4243
(Not all processes could be identified, non-owned process info
 will not be shown, you would have to be root to see it all.)
# 将docker设置为开机启动
gswewf@gswewf-pc:~$ sudo systemctl enable docker

