FROM alpine:3.8

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

# RUN echo `python --version`

ENV LANG zh_CN.UTF-8
ENV LANGUAGE zh_CN:zh
ENV LC_ALL zh_CN.UTF-8

WORKDIR /project

COPY requirements.txt /project/

RUN set -e && \
    apk update \
        --repository https://mirrors.aliyun.com/alpine/v3.8/main/ \
        --allow-untrusted && \
    apk add \
        # --no-cache \
        # --virtual .build-deps \
        # openssh-client \
        gcc \
        g++ \
        wget \
        curl \
        vim \
        bash \
        bash-doc \
        bash-completion \
        # pcre-dev \
        build-base \
        python3-dev \
        # libevent-dev \
        # cython \
        py3-pip \
        git \
        #py3-numpy \
        py3-scipy \
        --repository https://mirrors.aliyun.com/alpine/v3.8/main/ \
        --allow-untrusted \
    && \
    pip3 install -r requirements.txt -i http://pypi.douban.com/simple \
     --trusted-host=pypi.douban.com
    #&& apk del -f .build-deps


RUN echo "alias date='date +\"%Y-%m-%d %H:%M:%S\"'" >> ~/.bashrc

#COPY docker-entrypoint.sh /usr/local/bin/

EXPOSE 5001 9091

#CMD ["docker-entrypoint.sh"]


