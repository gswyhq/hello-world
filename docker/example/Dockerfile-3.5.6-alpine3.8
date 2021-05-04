FROM python:3.5.6-alpine3.8

# 修改时区
RUN apk update \
        --repository https://mirrors.aliyun.com/alpine/v3.8/main/ \
        --allow-untrusted && \
    apk --no-cache add tzdata --virtual abcdefg \
        --repository https://mirrors.aliyun.com/alpine/v3.8/main/ \
        --allow-untrusted  && \
    ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && \
    echo "Asia/Shanghai" > /etc/timezone && \
    apk del -f abcdefg

RUN apk add py3-lxml bash-completion colordiff vim

ENV LANG zh_CN.UTF-8
ENV LANGUAGE zh_CN:zh
ENV LC_ALL zh_CN.UTF-8

WORKDIR /usr/src/app

# COPY requirements.txt ./

RUN apk --update add --no-cache nginx
RUN mkdir /run/nginx
RUN apk --update add --no-cache \
    git

RUN apk --update add --no-cache --virtual .build-deps \
        gfortran \
        musl-dev \
        libxslt-dev \
        g++ \
        lapack-dev \
        gcc \
        freetype-dev \
        --repository http://mirrors.ustc.edu.cn/alpine/v3.8/main/ --allow-untrusted \
    && \
    pip install requests==2.12.4 && \
    pip install pandas==0.19.2 && \
    pip install pypinyin==0.23.0 && \
    pip install elasticsearch==5.3.0 && \
    pip install redis==2.10.5 && \
    pip install Cython==0.27.3 && \
    pip install tornado==4.4.2 && \
    pip install jieba==0.39 && \
    pip install py2neo==3.1.2 && \
    pip3 install git+https://github.com/Supervisor/supervisor.git \
    # pip3 --default-timeout=60000 install -r requirements.txt  \
    && apk del -f .build-deps

RUN git config --global core.quotepath false && \
    git config --global log.date iso8601 && \
    git config --global credential.helper store

RUN mkdir /var/log/supervisor && \
    mkdir -p /etc/supervisor/conf.d && \
    echo_supervisord_conf > /etc/supervisor/supervisord.conf && \
    echo "[include]" >> /etc/supervisor/supervisord.conf && \
    echo "files = /etc/supervisor/conf.d/*.conf" >> /etc/supervisor/supervisord.conf

RUN echo 'if [ "$color_prompt" = yes ]; then' >> ~/.bashrc
RUN echo "    PS1='${debian_chroot:+($debian_chroot)}\[\033[01;32m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ '" >> ~/.bashrc
RUN echo "else" >> ~/.bashrc
RUN echo "    PS1='${debian_chroot:+($debian_chroot)}\u@\h:\w\$ '" >> ~/.bashrc
RUN echo "fi" >> ~/.bashrc

RUN echo "alias date='date +\"%Y-%m-%d %H:%M:%S\"'" >> ~/.bashrc
RUN echo "export TERM=xterm" >> ~/.bashrc
RUN echo "source /usr/share/bash-completion/completions/git" >> ~/.bashrc
RUN echo "export PATH=/bin/bash:$PATH" >> ~/.bashrc

EXPOSE 8000

CMD [ "python3"]

# $ docker build -t alpine_20180919 -f Dockerfile .
# $ docker run -it --rm --name my-running-app alpine_20180919

