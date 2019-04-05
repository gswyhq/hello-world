#!/usr/bin/python3
# coding=utf-8

# 安装：
# gswyhq@gswyhq-PC:~/k8s/aliyuncs/k8s$ pip3 install PyYAML

import yaml

# 读文件
# 随便写个yaml的文件，比如 config.yaml：


# 然后解析它：


def read_yaml():
    with open('config.yaml', encoding='utf-8') as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
        print(config)

# 可以直接从文件加载，加载进来之后是一个字典，打印输出如下：


# {'apple': {'color': 'red', 'weight': 500}, 'dog': {'name': 'DaHuang'}}

# 写文件
def write_yaml():
    guy = {
        'name': '陈二',
        'age': '22',
        'tag': {'loser':3, "labels": 'app'}
    }

    # 直接dump可以把对象转为YAML文档
    print(yaml.dump(guy, allow_unicode=True))

    # 也可以直接dump到文件或者流中
    with open('config2.yaml', 'w', encoding='utf-8') as guy_file:
        yaml.dump(guy, guy_file, Dumper=yaml.Dumper, allow_unicode=True)

def main():
    read_yaml()
    write_yaml()

if __name__ == '__main__':
    main()