#!/bin/bash

input_host_port='192.168.3.105:9200'
output_host_port='192.168.3.145:9200'
index_name='xinxin_templates_question_20180309_151024'
index_name_alias='xinxin_templates_question_alias'

# 获取使用帮助：
#gswyhq@gswyhq-PC:~/yhb/es_search$ ./elasticdump.sh -h

# gswyhq@gswyhq-PC:~/yhb/es_search$ ./elasticdump.sh -i 192.168.3.105:9200 -o 192.168.3.145:9200 -n xinxin_templates_question_20180309_184614 -a xinxin_templates_question_alias

function usage() {
        echo "使用方法:"
        echo "  ./elasticdump.sh [-h] [-i <源数据地址，如：192.168.3.105:9200>] [-o <目标地址，如：52.80.187.77:18200>] [-n <待迁移es数据索引名>] [-a <新建的索引别名>]"
        exit 1
}

# 在 while 循环中使用 getopts 解析命令行选项
# 要解析的选项有 -h、-v、-f 和 -o，其中 -f 和 -o 选项带有参数
# 字符串选项中第一个冒号表示 getopts 使用抑制错误报告模式

while getopts :hi:o:n:a: opt
do
        case "$opt" in
                i)
                        input_host_port=$OPTARG
                        ;;
                o)
                        output_host_port=$OPTARG
                        ;;
                n)
                        index_name=$OPTARG
                        ;;
                a)
                        index_name_alias=$OPTARG
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

echo "${input_host_port} 上的索引 ${index_name} 迁移到 ${output_host_port}"

# 从一个机器迁移索引及数据到另一个机器:
echo "第一步： 拷贝analyzer如分词"
docker run --rm -ti taskrabbit/elasticsearch-dump   --input="http://${input_host_port}/${index_name}"   --output="http://${output_host_port}/${index_name}"   --type=analyzer

echo "第二步： 拷贝映射"
docker run --rm -ti taskrabbit/elasticsearch-dump   --input="http://${input_host_port}/${index_name}"   --output="http://${output_host_port}/${index_name}"   --type=mapping

echo "第三步： 拷贝数据"
docker run --rm -ti taskrabbit/elasticsearch-dump   --input="http://${input_host_port}/${index_name}"   --output="http://${output_host_port}/${index_name}"   --type=data

# 以上数据迁移的时候，会丢失对应的索引别名信息；

#若不需要删除原索引别名，仅仅添加一个索引别名：
#curl -XPOST "${output_host_port}/_aliases?pretty" -H 'Content-Type: application/json' -d'
#{
#    "actions": [
#        { "add":    { "index": "'${index_name}'", "alias": "'${index_name_alias}'" }
#        }
#    ]
#}
#'

# 需要注意的是，json数据里变量要用''括起来

echo "${input_host_port} 上的索引 ${index_name} 成功迁移到 ${output_host_port}"

# 若仅仅是删除一个索引的别名：
#curl -XPOST '192.168.3.105:9200/_aliases?pretty' -H 'Content-Type: application/json' -d'
#{
#    "actions": [
#        { "remove": { "index": "dingding_faq", "alias": "dingding_faq_alias" }
#        }
#    ]
#}
#'

# 检测别名`dingding_faq_alias`指向哪一个索引：
str=$(printf "%-150s" "别名指向：")
echo "${str// /-}"
echo "在 ${output_host_port} 索引别名 ${index_name_alias} 的指向结果： "
echo "curl -XGET '${output_host_port}/*/_alias/${index_name_alias}?pretty'"
curl -XGET "${output_host_port}/*/_alias/${index_name_alias}?pretty"

# 哪些别名指向索引`dingding_faq`：
#curl -XGET 'localhost:9200/dingding_faq/_alias/*?pretty'


str=$(printf "%-150s" "添加索引别名：")
echo "${str// /-}"

echo "curl -XPOST ${output_host_port}/_aliases?pretty -H Content-Type: application/json -d'"
echo '{'
echo '   "actions": ['
echo '         { "add":    { "index": "'${index_name}'", "alias": "'${index_name_alias}'" } '
echo '         }'
echo '      ]'
echo '   }'
echo "'"


str=$(printf "%-150s" "切换索引别名：")
echo "${str// /-}"
echo "curl -XPOST ${output_host_port}/_aliases?pretty -H Content-Type: application/json -d'"
echo '{'
echo '   "actions": ['
echo '         { "remove":    { "index": "旧的索引名", "alias": "'${index_name_alias}'" }}, '
echo '         { "add":    { "index": "'${index_name}'", "alias": "'${index_name_alias}'" }} '
echo '      ]'
echo '   }'
echo "'"

