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

#data_dir=data_`date +%s%N`
#
#mkdir ${data_dir}

if [[ ${input_host_port} != "http*" ]];
then
    input_host_port="http://${input_host_port}"
fi

if [[ ${output_host_port} != "http*" ]];
then
    output_host_port="http://${output_host_port}"
fi

echo "${input_host_port} 上的索引前缀为 ${pid_name} 迁移到 ${output_host_port}"

index_names=`curl ${input_host_port}/_cat/indices/${pid_name}*_alias -s |grep open|awk '{print $3}'`
#index_names=`curl 192.168.3.105:9200/_cat/indices/fdrs*_alias -s |grep open|awk '{print $3}'`

#添加一个参数-s，可以去除这些统计信息。(Curl不显示统计信息% Total % Received %)
#curl http://www.baidu.com -s

echo -e "\n将迁移如下索引:\n $index_names"

read -r -p "是否迁移以上索引? [Y/n] " input

case $input in
    [yY][eE][sS]|[yY])
		echo "开始迁移数据"
		;;

    [nN][oO]|[nN])
		echo "退出"
		exit 0
       	;;

    *)
	echo "无效的输入值..."
	exit 1
	;;
esac

function es_index_dump(){
#    echo "这是我的第一个 shell 函数!"
    local index_name=${1}
    docker run --rm -ti taskrabbit/elasticsearch-dump:v3.3.14 /bin/multielasticdump \
        --input=${input_host_port} --input-index=${index_name} \
        --output=${output_host_port} --output-index=${index_name} \
        --type=settings

    docker run --rm -ti taskrabbit/elasticsearch-dump:v3.3.14 /bin/multielasticdump \
        --input=${input_host_port} --input-index=${index_name} \
        --output=${output_host_port} --output-index=${index_name} \
        --type=mapping

    docker run --rm -ti taskrabbit/elasticsearch-dump:v3.3.14 /bin/multielasticdump \
        --input=${input_host_port} --input-index=${index_name} \
        --output=${output_host_port} --output-index=${index_name} \
        --type=data

#    docker run --rm -ti taskrabbit/elasticsearch-dump:v3.3.14 /bin/multielasticdump \
#        --input=${input_host_port} --input-index=${index_name} \
#        --output=${output_host_port} --output-index=${index_name} \
#        --type=alias

    index_alias=${index_name/%[0-9]*/alias};

    # 若原有别名存在，则删除原有别名，在新建别名；若不存在，则直接添加
    result_code=`curl -I -m 10 -o /dev/null -s -w %{http_code} "${output_host_port}/*/_alias/${index_alias}"`;

    if [ ${result_code} == 200 ];then
        curl -XPOST "${output_host_port}/_aliases" -H 'Content-Type: application/json' -d '
            {
                "actions": [
                    {"remove": {"index": "*", "alias": "'${index_alias}'"}},
                    {"add": {"index":"'${index_name}'", "alias": "'${index_alias}'"}}
                ]
            }';
    else
        curl -X POST "${output_host_port}/_aliases" -H 'Content-Type: application/json' -d'
            {
                "actions" : [
                    { "add" : { "index" : "'${index_name}'", "alias" : "'${index_alias}'" } }
                ]
            }';

    fi

    return 0;
    #    参数返回，可以显示加：return 返回，如果不加，将以最后一条命令运行结果作为返回值。
    #shell函数返回值是整形，并且在0~257之间。
}

for index in $index_names;
    do
        echo "导出 ${input_host_port}/${index}";
        es_index_dump ${index};
    done

#echo "导出 ${input_host_port}/${pid_name}* 数据到临时目录： ${data_dir}"

# 从一个机器迁移索引及数据到另一个机器:
#docker run --rm -ti taskrabbit/elasticsearch-dump:v3.3.14 /bin/multielasticdump --input="http://42.93.187.77:18200" --input-index="fdrs_templates_answer_20180717_094549" --output="http://192.168.3.145:9200" --output-index=fdrs_templates_answer_20180717_094549 --type=settings
#docker run --rm -ti taskrabbit/elasticsearch-dump:v3.3.14 /bin/multielasticdump --input="http://42.93.187.77:18200" --input-index="fdrs_templates_answer_20180717_094549" --output="http://192.168.3.145:9200" --output-index=fdrs_templates_answer_20180717_094549 --type=mapping
#docker run --rm -ti taskrabbit/elasticsearch-dump:v3.3.14 /bin/multielasticdump --input="http://42.93.187.77:18200" --input-index="fdrs_templates_answer_20180717_094549" --output="http://192.168.3.145:9200" --output-index=fdrs_templates_answer_20180717_094549 --type=data
#docker run --rm -ti taskrabbit/elasticsearch-dump:v3.3.14 /bin/multielasticdump --input="http://42.93.187.77:18200" --input-index="fdrs_templates_answer_20180717_094549" --output="http://192.168.3.145:9200" --output-index=fdrs_templates_answer_20180717_094549 --type=alias

echo "${input_host_port} 上的索引前缀 ${pid_name} 成功迁移到 ${output_host_port}"

# 哪些别名指向索引`dingding_faq`：
#curl -XGET '42.93.171.44:8861/dingding_faq/_alias/*?pretty'

# gswyhq@gswyhq-PC:~/yhb$ docker run --rm -ti -v $PWD/data:/data taskrabbit/elasticsearch-dump:v3.3.14 /bin/multielasticdump --input="http://192.168.3.105:9200/all_baoxian*" --output=/data/alias.json --type=alias
# elasticdump --input=./alias.json --output=http://es.com:9200 --type=alias

# curl -XGET 42.93.187.77:18200/jrtz_kg_entity_synonyms_alias/_count?
# curl -XGET -u elastic:web12008 '192.168.3.105:9200/_cat/indices/all_baoxian*'

# 42.93.171.44:8861
