#!/bin/bash


echo "项目id: $1"
pid=$1

if [ ! $pid ]
then
    echo "请在参数中输入项目id, 如： ./es_neo4j_data_package.sh xinxin_yyb"
    exit 1
fi

curl -sSL http://54.223.108.99:8811/api/get_token|sh

# 打包es数据

# 通过自定义镜像启动一个容器：
temp_es_data_container_name=${pid}_temp_es_container_`date +%Y%m%d_%H%M%S`
echo "构建临时es容器：${temp_es_data_container_name}"
docker run -d --name=${temp_es_data_container_name} \
        -p 9205:9200 \
        -v /etc/localtime:/etc/localtime \
        gswyhq/elasticsearch-ce-alpine:v5.6.10

sleep 30

# 通过脚本迁移数据：
echo "向临时容器：${temp_es_data_container_name}，写数据"
echo "y" |./multi_elasticdump.sh -i 192.168.3.105:9200 -o 192.168.3.105:9205 -p ${pid}_kg_entity_synonyms

# 创建包含数据的镜像:
es_data_image_version=${pid}_es_data_`date +%Y%m%d_%H%M%S`
docker commit -m "${pid}项目，es数据" ${temp_es_data_container_name}  899150993273.dkr.ecr.cn-north-1.amazonaws.com.cn/nlp:${es_data_image_version}

echo "临时es容器：${temp_es_data_container_name}，commit成镜像：899150993273.dkr.ecr.cn-north-1.amazonaws.com.cn/nlp:${es_data_image_version}"
echo "es数据镜像版本： 899150993273.dkr.ecr.cn-north-1.amazonaws.com.cn/nlp:${es_data_image_version}" >> es数据镜像构建记录.log

docker stop ${temp_es_data_container_name}
docker rm -v ${temp_es_data_container_name}

echo "将镜像： 899150993273.dkr.ecr.cn-north-1.amazonaws.com.cn/nlp:${es_data_image_version}， push到服务器"
docker push 899150993273.dkr.ecr.cn-north-1.amazonaws.com.cn/nlp:${es_data_image_version}
docker rmi 899150993273.dkr.ecr.cn-north-1.amazonaws.com.cn/nlp:${es_data_image_version}

# 打包neo4j数据
# 通过自定义镜像启动一个容器：
temp_neo4j_data_container_name=${pid}_temp_neo4j_container_`date +%Y%m%d_%H%M%S`

docker run --detach --name=${temp_neo4j_data_container_name} \
        -e TZ='CST-8' \
        -e NEO4J_AUTH=neo4j/gswyhq \
        -e NEO4J_dbms_security_procedures_unrestricted=apoc.\\\* \
        -e NEO4J_apoc_export_file_enabled=true \
        -e NEO4J_apoc_import_file_enabled=true \
        --env NEO4J_dbms_allow__format__migration=true \
        -e NEO4J_dbms_shell_enabled=true \
        -e NEO4J_dbms_shell_host=0.0.0.0 \
        -e NEO4J_dbms_shell_port=1337 \
        -e NEO4J_dbms_active__database=graph.db \
        gswyhq/neo4j-ce-alpine:3.4.5

echo "创建临时的neo4j数据容器：${temp_neo4j_data_container_name}"
sleep 15

# 导出数据到文件：
index_end=`date +%Y%m%d_%H%M%S`
cypher_data_file_name=${pid}_all_${index_end}.cypher
dump_cypher='CALL apoc.export.cypher.all(\"/var/lib/neo4j/data/'${cypher_data_file_name}'\",{})'

if [ $pid == "all_baoxian" ]
then
    curl http://192.168.3.105:47474/db/data/transaction/commit -u neo4j:gswyhq -H "Content-Type: application/json"  -d '{"statements" : [ {    "statement" : "'"${dump_cypher}"'" } ]}'
    echo "导出容器：xinxin_zhongjixian47474 的数据到文件: ${cypher_data_file_name}"
    docker cp xinxin_zhongjixian47474:/data/${cypher_data_file_name} .
elif [ $pid == "xinxin_yyb" ]
then
    curl http://192.168.3.105:17474/db/data/transaction/commit -u neo4j:gswyhq -H "Content-Type: application/json"  -d '{"statements" : [ {    "statement" : "'"${dump_cypher}"'" } ]}'
    echo "导出容器：xinxin_yyb_17474 的数据到文件: ${cypher_data_file_name}"
    docker cp xinxin_yyb_17474:/data/${cypher_data_file_name} .
else
    echo "项目id不应该为："${pid}
    docker stop ${temp_neo4j_data_container_name}
    docker rm -v ${temp_neo4j_data_container_name}
    exit 1
fi

echo "拷贝文件：${cypher_data_file_name} 到容器：${temp_neo4j_data_container_name}中 "
docker cp ${cypher_data_file_name} ${temp_neo4j_data_container_name}:/data/
rm ${cypher_data_file_name}

# 往容器中写入数据
echo "通过文件： ${cypher_data_file_name}，导入数据到容器： ${temp_neo4j_data_container_name}"
docker exec ${temp_neo4j_data_container_name} /var/lib/neo4j/bin/neo4j-shell -file /var/lib/neo4j/data/${cypher_data_file_name}

# 创建包含数据的镜像

neo4j_data_image_version=${pid}_neo4j_data_`date +%Y%m%d_%H%M%S`
echo "将容器：${temp_neo4j_data_container_name}， commit成镜像： 899150993273.dkr.ecr.cn-north-1.amazonaws.com.cn/nlp:${neo4j_data_image_version}"
docker commit -m "${pid}项目，neo4j数据" ${temp_neo4j_data_container_name}  899150993273.dkr.ecr.cn-north-1.amazonaws.com.cn/nlp:${neo4j_data_image_version}

docker stop ${temp_neo4j_data_container_name}
docker rm -v ${temp_neo4j_data_container_name}

echo "neo4j数据镜像版本: 899150993273.dkr.ecr.cn-north-1.amazonaws.com.cn/nlp:${neo4j_data_image_version}" >> neo4j数据镜像构建记录.log
echo "将镜像：899150993273.dkr.ecr.cn-north-1.amazonaws.com.cn/nlp:${neo4j_data_image_version}， push到服务器"
docker push 899150993273.dkr.ecr.cn-north-1.amazonaws.com.cn/nlp:${neo4j_data_image_version}
docker rmi 899150993273.dkr.ecr.cn-north-1.amazonaws.com.cn/nlp:${neo4j_data_image_version}



