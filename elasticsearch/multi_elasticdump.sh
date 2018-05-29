#!/bin/bash

input_host_port='192.168.3.105:9200'
output_host_port='192.168.3.145:9200'
pid_name='unknown'


# 获取使用帮助：
#gswyhq@gswyhq-PC:~/yhb/es_search$ ./elasticdump.sh -h

# gswyhq@gswyhq-PC:~/yhb/es_search$ ./elasticdump.sh -i 192.168.3.105:9200 -o 192.168.3.145:9200 -p xinxin

function usage() {
        echo "使用方法:"
        echo "  ./elasticdump.sh [-h] [-i <源数据地址，如：192.168.3.105:9200>] [-o <目标地址，如：elastic:web12008@42.93.187.77:18200>] [-p <待迁移es数据索引名前缀>]"
        exit 1
}

# 在 while 循环中使用 getopts 解析命令行选项
# 要解析的选项有 -h、-v、-f 和 -o，其中 -f 和 -o 选项带有参数
# 字符串选项中第一个冒号表示 getopts 使用抑制错误报告模式

while getopts :hi:o:p: opt
do
        case "$opt" in
                i)
                        input_host_port=$OPTARG
                        ;;
                o)
                        output_host_port=$OPTARG
                        ;;
                p)
                        pid_name=$OPTARG
                        ;;
                h)
                        usage
                        exit
                        ;;
                :)
                        echo "选项 -$OPTARG 需要有个参数."
                        usage
                        exit 1
                        ;;
                ?)
                        echo "无效的选项: -$OPTARG"
                        usage
                        exit 2
                        ;;
        esac
done

if [ $# -le 0 ];then
    echo "输入命令应该带有选项及参数！ "
    usage
    exit
else
    echo "输入的参数个数： $#"
fi

data_dir=data_`date +%s%N`

mkdir ${data_dir}

echo "${input_host_port} 上的索引前缀为 ${pid_name} 迁移到 ${output_host_port}"

echo "导出 ${input_host_port}/${pid_name}* 数据到临时目录： ${data_dir}"

# 从一个机器迁移索引及数据到另一个机器:

echo "第一步： 拷贝settings到临时目录"
docker run --rm -ti -v "$PWD/${data_dir}:/data" taskrabbit/elasticsearch-dump:v3.3.14 /bin/multielasticdump --input="http://${input_host_port}/${pid_name}*" --output="/data/${pid_name}_settings.json"   --type=settings

echo "第二步： 拷贝数据到临时目录"
# 每个匹配的索引都会创建 一个 data, mapping, 和 analyzer 文件
docker run --rm -ti -v "$PWD/${data_dir}:/data" taskrabbit/elasticsearch-dump:v3.3.14 /bin/multielasticdump --input="http://${input_host_port}/${pid_name}*" --output="/data/${pid_name}_data.json"  # --type=data

echo "第三步： 拷贝索引别名到临时目录"
docker run --rm -ti -v "$PWD/${data_dir}:/data" taskrabbit/elasticsearch-dump:v3.3.14 /bin/multielasticdump --input="http://${input_host_port}/${pid_name}*" --output="/data/${pid_name}_alias.json"   --type=alias


echo "从临时目录： ${data_dir}， 导入数据到 ${output_host_port}/${pid_name}* "

echo "第四步： 从临时目录拷贝settings"
docker run --rm -ti -v "$PWD/${data_dir}:/data" taskrabbit/elasticsearch-dump:v3.3.14 /bin/multielasticdump --direction load --output="http://${output_host_port}" --input="/data/${pid_name}_settings.json"   --type=settings

echo "第五步： 从临时目录拷贝数据"
docker run --rm -ti -v "$PWD/${data_dir}:/data" taskrabbit/elasticsearch-dump:v3.3.14 /bin/multielasticdump --direction load --output="http://${output_host_port}" --input="/data/${pid_name}_data.json"  # --type=data

curl -XDELETE ${output_host_port}/${pid_name}*_alias/_alias/${pid_name}*

echo "第六步： 从临时目录拷贝索引别名"
docker run --rm -ti -v "$PWD/${data_dir}:/data" taskrabbit/elasticsearch-dump:v3.3.14 /bin/multielasticdump --direction load --output="http://${output_host_port}" --input="/data/${pid_name}_alias.json"   --type=alias

rm -rf ${data_dir}

# 需要注意的是，json数据里变量要用''括起来

echo "${input_host_port} 上的索引前缀 ${pid_name} 成功迁移到 ${output_host_port}"

# 哪些别名指向索引`dingding_faq`：
#curl -XGET '42.93.171.45:8861/dingding_faq/_alias/*?pretty'

# gswyhq@gswyhq-PC:~/yhb$ docker run --rm -ti -v $PWD/data:/data taskrabbit/elasticsearch-dump:v3.3.14 /bin/multielasticdump --input="http://192.168.3.105:9200/all_baoxian*" --output=/data/alias.json --type=alias
# elasticdump --input=./alias.json --output=http://es.com:9200 --type=alias

# curl -XGET 42.93.187.77:18200/jrtz_kg_entity_synonyms_alias/_count?
# curl -XGET -u elastic:web12008 '192.168.3.105:9200/_cat/indices/all_baoxian*'

# 42.93.171.45:8861
