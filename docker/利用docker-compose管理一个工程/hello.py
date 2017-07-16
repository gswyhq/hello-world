#!/usr/bin/python3
# coding: utf-8
import os
import argparse
from flask import Flask
from redis import Redis

MYSQL_HOST = os.getenv('MYSQL_HOST')
MYSQL_PORT = os.getenv('MYSQL_PORT')

parser = argparse.ArgumentParser()
parser.add_argument('-qa', '--qa', dest='qa', action='store_true', help='启用用户自定义的配置（默认启用用户自定义配置）')
parser.add_argument('-no-qa', '-not-qa', '--not-qa', '--no-qa', dest='qa', action='store_false', help='不启用用户自定义的配置（默认启用用户自定义配置）')
parser.set_defaults(qa=True)

args = parser.parse_args()

print("命令行启动参数: {}".format(vars(args)))

app = Flask(__name__)
redis = Redis(host='docker-redis', port=6379)

@app.route('/')
def hello():
    redis.incr('hits')
    # Redis INCR命令用于将键的整数值递增1。如果键不存在，则在执行操作之前将其设置为0。
    # 如果键包含错误类型的值或包含无法表示为整数的字符串，则会返回错误。
    return '你好，这是第 %s 次请求.' % redis.get('hits')

def main():
    print('你好')
    print("环境变量`MYSQL_HOST`的值：", MYSQL_HOST)
    print("环境变量`MYSQL_PORT`的数据类型：", type(MYSQL_PORT))
    app.run(host="0.0.0.0", port=8000, debug=True)

if __name__ == '__main__':
    main()
