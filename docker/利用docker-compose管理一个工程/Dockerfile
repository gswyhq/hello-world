FROM ubuntu:0716

# 设置环境变量
ENV LANG zh_CN.UTF-8
ENV MYSQL_HOST localhost
ENV MYSQL_PORT 3306

# ADD . /src
ADD hello.py /src/hello.py

# WORKDIR: 当前的工作目录
WORKDIR /src/

EXPOSE 8000

# 设置container启动时执行的操作，指定了该镜像启动时的入口
# ENTRYPOINT ["python3", "/src/tornado_server.py"]
ENTRYPOINT ["python3", "hello.py"]

# 指定了容器启动时的命令
# CMD ["--templates", "--no-classified"]
CMD ["-qa"]

