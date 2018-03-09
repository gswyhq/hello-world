#!/bin/bash

input_host_port='192.168.3.105:9200'
output_host_port='192.168.3.145:9200'
index_name='xinxin_templates_question_20180309_151024'
index_name_alias='xinxin_templates_question_alias'

# 从一个机器迁移索引及数据到另一个机器:
echo "第一步： 拷贝analyzer如分词"
#docker run --rm -ti taskrabbit/elasticsearch-dump   --input="http://${input_host_port}/${index_name}"   --output="http://${output_host_port}/${index_name}"   --type=analyzer

echo "第二步： 拷贝映射"
#docker run --rm -ti taskrabbit/elasticsearch-dump   --input="http://${input_host_port}/${index_name}"   --output="http://${output_host_port}/${index_name}"   --type=mapping

echo "第三步： 拷贝数据"
#docker run --rm -ti taskrabbit/elasticsearch-dump   --input="http://${input_host_port}/${index_name}"   --output="http://${output_host_port}/${index_name}"   --type=data

# 以上数据迁移的时候，会丢失对应的索引别名信息；

#若不需要删除原索引别名，仅仅添加一个索引别名：
curl -XPOST "${output_host_port}/_aliases?pretty" -H 'Content-Type: application/json' -d'
{
    "actions": [
        { "add":    { "index": "'${index_name}'", "alias": "'${index_name_alias}'" }
        }
    ]
}
'

# 需要注意的是，json数据里变量要用''括起来

echo "${input_host_port} 上的索引 ${index_name} 迁移到 ${output_host_port}"

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
#curl -XGET 'localhost:9200/*/_alias/dingding_faq_alias?pretty'

# 哪些别名指向索引`dingding_faq`：
#curl -XGET 'localhost:9200/dingding_faq/_alias/*?pretty'
