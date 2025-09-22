


1、开发环境如何部署？
答：git clone https://github.com/coze-dev/coze-studio 
cd docker
docker compose -f docker-compose.yml up -d

若不能使用docker compose,只有docker只能通过如下步骤：
第一步，准备docker镜像文件
第二步：导入镜像
docker load < elasticsearch_8.18.0-save.tgz;
docker load < minio_RELEASE.2025-06-13T11-33-47Z-cpuv1-save.tgz;
docker load < opencoze_latest-save.tgz;
docker load < etcd_3.5-save.tgz;
docker load < mysql_8.4.5-save.tgz;
docker load < redis_8.0-save.tgz;
docker load < milvus_v2.5.10-save.tgz;
docker load < nsq_v1.2.1-save.tgz; 


第三步：# 创建自定义网络
docker network create coze-network

第四步：准备，需要挂载的目录
(DEV)[root@zhangsan coze-studio]# cp -r coze-studio/docker/volumes .
(DEV)[root@zhangsan coze-studio]# pwd
/home/ecsuser/chatbi/coze-studio 


第五步：准备环境变量
(DEV)[ecsuser@zhangsan coze-studio]$ cp coze-studio/docker/.env .
# 加载 .env 文件
source .env

# 使用 envsubst 展开变量
envsubst < .env > .env.expanded

# 删除 export（如果还存在）
sed 's/export //g' -i .env.expanded 

# 删除双引号
sed 's/"//g' -i .env.expanded  
# 手动将变量值后面的#注释部分移到下一行

#完整.env.expanded  文件内容：
(DEV)[ecsuser@zhangsan coze-studio]$ cat .env.expanded
# Server
LISTEN_ADDR=0.0.0.0:8888
LOG_LEVEL=debug
MAX_REQUEST_BODY_SIZE=1073741824
SERVER_HOST=localhost0.0.0.0:8888
MINIO_PROXY_ENDPOINT=:8889
USE_SSL=0
SSL_CERT_FILE=
SSL_KEY_FILE=
# MySQL
MYSQL_ROOT_PASSWORD=root
MYSQL_DATABASE=opencoze
MYSQL_USER=coze
MYSQL_PASSWORD=coze123
MYSQL_HOST=coze-mysql
MYSQL_PORT=3306
MYSQL_DSN=coze:coze123@tcp(coze-mysql:3306)/opencoze?charset=utf8mb4&parseTime=True
ATLAS_URL=mysql://coze:coze123@coze-mysql:3306/opencoze?charset=utf8mb4&parseTime=True
# Redis
REDIS_AOF_ENABLED=no
REDIS_IO_THREADS=4
ALLOW_EMPTY_PASSWORD=yes
REDIS_ADDR=coze-redis:6379
REDIS_PASSWORD=
# This Upload component used in Agent / workflow File/Image With LLM  , support the component of imagex / storage
# default: storage, use the settings of storage component
# if imagex, you must finish the configuration of <VolcEngine ImageX>
FILE_UPLOAD_COMPONENT_TYPE=storage
# VolcEngine ImageX
VE_IMAGEX_AK=
VE_IMAGEX_SK=
VE_IMAGEX_SERVER_ID=
VE_IMAGEX_DOMAIN=
VE_IMAGEX_TEMPLATE=
VE_IMAGEX_UPLOAD_HOST=https://imagex.volcengineapi.com
# Storage component
STORAGE_TYPE=minio
# minio / tos / s3
STORAGE_UPLOAD_HTTP_SCHEME=http
# http / https. If coze studio website is https, you must set it to https
STORAGE_BUCKET=opencoze
# MiniIO
MINIO_ROOT_USER=minioadmin
MINIO_ROOT_PASSWORD=minioadmin123
MINIO_DEFAULT_BUCKETS=milvus
MINIO_AK=minioadmin
MINIO_SK=minioadmin123
MINIO_ENDPOINT=coze-minio:9000
MINIO_API_HOST=http://coze-minio:9000
# TOS
TOS_ACCESS_KEY=
TOS_SECRET_KEY=
TOS_ENDPOINT=https://tos-cn-beijing.volces.com
TOS_BUCKET_ENDPOINT=https://opencoze.tos-cn-beijing.volces.com
TOS_REGION=cn-beijing
# S3
S3_ACCESS_KEY=
S3_SECRET_KEY=
S3_ENDPOINT=
S3_BUCKET_ENDPOINT=
S3_REGION=
# Elasticsearch
ES_ADDR=http://coze-elasticsearch:9200
ES_VERSION=v8
ES_USERNAME=
ES_PASSWORD=
COZE_MQ_TYPE=nsq
# nsq / kafka / rmq
MQ_NAME_SERVER=coze-nsqd:4150
# RocketMQ
RMQ_ACCESS_KEY=
RMQ_SECRET_KEY=
# Settings for VectorStore
# VectorStore type: milvus / vikingdb
# If you want to use vikingdb, you need to set up the vikingdb configuration.
VECTOR_STORE_TYPE=milvus
# milvus vector store
MILVUS_ADDR=coze-milvus:19530
# vikingdb vector store for Volcengine
VIKING_DB_HOST=
VIKING_DB_REGION=
VIKING_DB_AK=
VIKING_DB_SK=
VIKING_DB_SCHEME=
VIKING_DB_MODEL_NAME=
# if vikingdb model name is not set, you need to set Embedding settings
# Settings for Embedding
# The Embedding model relied on by knowledge base vectorization does not need to be configured
# if the vector database comes with built-in Embedding functionality (such as VikingDB). Currently,
# Coze Studio supports three access methods: openai, ark, ollama, and custom http. Users can simply choose one of them when using
# embedding type: openai / ark / ollama / http
EMBEDDING_TYPE=openai
EMBEDDING_MAX_BATCH_SIZE=50
# openai embedding
OPENAI_EMBEDDING_BASE_URL=http://192.168.82.29:7866/compatible-mode/v1
# (string) OpenAI base_url
OPENAI_EMBEDDING_MODEL=bge-small-zh-v1.5
# (string) OpenAI embedding model
OPENAI_EMBEDDING_API_KEY=emb-9sF!2Lm@qW7xZ423we23e
# (string) OpenAI api_key
OPENAI_EMBEDDING_BY_AZURE=false
# (bool) OpenAI by_azure
OPENAI_EMBEDDING_API_VERSION=v1
# OpenAI azure api version
OPENAI_EMBEDDING_DIMS=512
# (int) 向量维度
OPENAI_EMBEDDING_REQUEST_DIMS=1024
# ark embedding
ARK_EMBEDDING_MODEL=bge-small-zh-v1.5
ARK_EMBEDDING_API_KEY=emb-9sF!2Lm@qW7xZ423we23e
ARK_EMBEDDING_DIMS=512
ARK_EMBEDDING_BASE_URL=http://192.168.82.29:7866/compatible-mode/v1/embeddings
# ollama embedding
OLLAMA_EMBEDDING_BASE_URL=
OLLAMA_EMBEDDING_MODEL=
OLLAMA_EMBEDDING_DIMS=
# http embedding
HTTP_EMBEDDING_ADDR=
HTTP_EMBEDDING_DIMS=1024
# Settings for OCR
# If you want to use the OCR-related functions in the knowledge base feature，You need to set up the OCR configuration.
# Currently, Coze Studio has built-in Volcano OCR.
# ocr_type: default type `ve`
OCR_TYPE=ve
# ve ocr
VE_OCR_AK=
VE_OCR_SK=
# Settings for Model
# Model for agent & workflow
# add suffix number to add different models
MODEL_PROTOCOL_0=ark
# protocol
MODEL_OPENCOZE_ID_0=100001
# id for record
MODEL_NAME_0=
# model name for show
MODEL_ID_0=
# model name for connection
MODEL_API_KEY_0=
# model api key
MODEL_BASE_URL_0=
# model base url
# Model for knowledge nl2sql, messages2query (rewrite), image annotation, workflow knowledge recall
# add prefix to assign specific model, downgrade to default config when prefix is not configured:
# 1. nl2sql:                    NL2SQL_ (e.g. NL2SQL_BUILTIN_CM_TYPE)
# 2. messages2query:            M2Q_    (e.g. M2Q_BUILTIN_CM_TYPE)
# 3. image annotation:          IA_     (e.g. IA_BUILTIN_CM_TYPE)
# 4. workflow knowledge recall: WKR_    (e.g. WKR_BUILTIN_CM_TYPE)
# supported chat model type: openai / ark / deepseek / ollama / qwen / gemini
BUILTIN_CM_TYPE=ark
# type openai
BUILTIN_CM_OPENAI_BASE_URL=
BUILTIN_CM_OPENAI_API_KEY=
BUILTIN_CM_OPENAI_BY_AZURE=false
BUILTIN_CM_OPENAI_MODEL=
# type ark
BUILTIN_CM_ARK_API_KEY=
BUILTIN_CM_ARK_MODEL=
BUILTIN_CM_ARK_BASE_URL=
# type deepseek
BUILTIN_CM_DEEPSEEK_BASE_URL=
BUILTIN_CM_DEEPSEEK_API_KEY=
BUILTIN_CM_DEEPSEEK_MODEL=
# type ollama
BUILTIN_CM_OLLAMA_BASE_URL=
BUILTIN_CM_OLLAMA_MODEL=
# type qwen
BUILTIN_CM_QWEN_BASE_URL=
BUILTIN_CM_QWEN_API_KEY=
BUILTIN_CM_QWEN_MODEL=
# type gemini
BUILTIN_CM_GEMINI_BACKEND=
BUILTIN_CM_GEMINI_API_KEY=
BUILTIN_CM_GEMINI_PROJECT=
BUILTIN_CM_GEMINI_LOCATION=
BUILTIN_CM_GEMINI_BASE_URL=
BUILTIN_CM_GEMINI_MODEL=
# Workflow Code Runner Configuration
# Supported code runner types: sandbox / local
# Default using local
# - sandbox: execute python code in a sandboxed env with deno + pyodide
# - local: using venv, no env isolation
CODE_RUNNER_TYPE=local
# Sandbox sub configuration
# Access restricted to specific environment variables, split with comma, e.g. PATH,USERNAME
CODE_RUNNER_ALLOW_ENV=
# Read access restricted to specific paths, split with comma, e.g. /tmp,./data
CODE_RUNNER_ALLOW_READ=
# Write access restricted to specific paths, split with comma, e.g. /tmp,./data
CODE_RUNNER_ALLOW_WRITE=
# Subprocess execution restricted to specific commands, split with comma, e.g. python,git
CODE_RUNNER_ALLOW_RUN=
# Network access restricted to specific domains/IPs, split with comma, e.g. api.test.com,api.test.org:8080
# The following CDN supports downloading the packages required for pyodide to run Python code. Sandbox may not work properly if removed.
CODE_RUNNER_ALLOW_NET=cdn.jsdelivr.net
# Foreign Function Interface access to specific libraries, split with comma, e.g. /usr/lib/libm.so
CODE_RUNNER_ALLOW_FFI=
# Directory for deno modules, default using pwd. e.g. /tmp/path/node_modules
CODE_RUNNER_NODE_MODULES_DIR=
# Code execution timeout, default 60 seconds. e.g. 2.56
CODE_RUNNER_TIMEOUT_SECONDS=
# Code execution memory limit, default 100MB. e.g. 256
CODE_RUNNER_MEMORY_LIMIT_MB=
# The function of registration controller
# If you want to disable the registration feature, set DISABLE_USER_REGISTRATION to true. You can then control allowed registrations via a whitelist with ALLOW_REGISTRATION_EMAIL.
DISABLE_USER_REGISTRATION=
# default , if you want to disable, set to true
ALLOW_REGISTRATION_EMAIL=
#  is a list of email addresses, separated by ,. Example: 11@example.com,22@example.com
第六步：配置大模型

答：问答大模型配置如下，但知识库用到的向量搜索模型，需要在上面环境变量部分配置，如EMBEDDING_TYPE等环境变量

(DEV)[ecsuser@zhangsan model]$ pwd
/home/ecsuser/chatbi/coze-studio/coze_conf/model
(DEV)[ecsuser@zhangsan model]$ cp template/model_template_openai.yaml open.yaml
修改name、base_url
name: OpenGPT-Pro4.0
(DEV)[ecsuser@zhangsan model]$ grep base_url open.yaml
        base_url: "http://192.168.82.29:8827/v2"
第七步：按以下顺序启动docker容器：
mysql
redis
elasticsearch
minio
etcd
milvus
nsqlookupd
nsqd
nsqadmin
coze-server

 1. MySQL
docker run -d --privileged=true \
  --name coze-mysql \
  --network coze-network \
  -p 3306:3306 \
  -v $PWD/data/mysql:/var/lib/mysql \
  -v $PWD/volumes/mysql/schema.sql:/docker-entrypoint-initdb.d/init.sql \
  -e MYSQL_ROOT_PASSWORD=${MYSQL_ROOT_PASSWORD:-root} \
  -e MYSQL_DATABASE=${MYSQL_DATABASE:-opencoze} \
  -e MYSQL_USER=${MYSQL_USER:-coze} \
  -e MYSQL_PASSWORD=${MYSQL_PASSWORD:-coze123} \
  --health-cmd="mysqladmin ping -h localhost -u\$${MYSQL_USER} -p\$${MYSQL_PASSWORD}" \
  --health-interval=10s \
  --health-timeout=5s \
  --health-retries=5 \
  --health-start-period=60s \
  mysql:8.4.5 \
  --character-set-server=utf8mb4 \
  --collation-server=utf8mb4_unicode_ci

2. Redis
docker run -d \
  --name coze-redis \
  --network coze-network \
  -p 6379:6379 \
  -v $PWD/data/bitnami/redis:/bitnami/redis/data:rw,Z \
  -e REDIS_AOF_ENABLED=${REDIS_AOF_ENABLED:-no} \
  -e REDIS_PORT_NUMBER=${REDIS_PORT_NUMBER:-6379} \
  -e REDIS_IO_THREADS=${REDIS_IO_THREADS:-4} \
  -e ALLOW_EMPTY_PASSWORD=${ALLOW_EMPTY_PASSWORD:-yes} \
  --health-cmd="redis-cli ping" \
  --health-interval=5s \
  --health-timeout=10s \
  --health-retries=10 \
  --health-start-period=60s \
  --privileged \
  bitnami/redis:8.0

3. Elasticsearch
docker run -d \
  --name coze-elasticsearch \
  --restart always \
  --user root \
  --privileged \
  -e TEST=1 \
  -e ES_ADDR=http://coze-elasticsearch:9200 \
  -p 19200:9200 \
  -v $PWD/data/bitnami/elasticsearch:/bitnami/elasticsearch/data \
  -v $PWD/volumes/elasticsearch/elasticsearch.yml:/opt/bitnami/elasticsearch/config/my_elasticsearch.yml \
  -v $PWD/volumes/elasticsearch/analysis-smartcn.zip:/opt/bitnami/elasticsearch/analysis-smartcn.zip:rw,Z \
  -v $PWD/volumes/elasticsearch/setup_es.sh:/setup_es.sh \
  -v $PWD/volumes/elasticsearch/es_index_schema:/es_index_schemas \
  --network coze-network \
  --health-cmd "curl -f http://localhost:9200 || exit 1" \
  --health-interval 15s \
  --health-timeout 10s \
  --health-retries 20 \
  --health-start-period 160s \
  bitnami/elasticsearch:8.18.0 \
  bash -c "
    /opt/bitnami/scripts/elasticsearch/setup.sh;
    chown -R elasticsearch:elasticsearch /bitnami/elasticsearch/data;
    chmod g+s /bitnami/elasticsearch/data;
    mkdir -p /bitnami/elasticsearch/plugins;
    echo 'Installing smartcn plugin...';
    if [ ! -d /opt/bitnami/elasticsearch/plugins/analysis-smartcn ]; then
      echo 'Copying smartcn plugin...';
      cp /opt/bitnami/elasticsearch/analysis-smartcn.zip /tmp/analysis-smartcn.zip;
      elasticsearch-plugin install file:///tmp/analysis-smartcn.zip;
      if [[ \"\$?\" != \"0\" ]]; then
        echo 'Plugin installation failed, exiting operation';
        rm -rf /opt/bitnami/elasticsearch/plugins/analysis-smartcn;
        exit 1;
      fi;
      rm -f /tmp/analysis-smartcn.zip;
    fi;
    touch /tmp/es_plugins_ready;
    echo 'Plugin installation successful, marker file created';
    (
      echo 'Waiting for Elasticsearch to be ready...'
      until curl -s -f http://localhost:9200/_cat/health > /dev/null 2>&1; do
        echo 'Elasticsearch not ready, waiting...'
        sleep 2;
      done;
      echo 'Elasticsearch is ready!';
      echo 'Running Elasticsearch initialization...'
      sed 's/\r$$//' /setup_es.sh > /setup_es_fixed.sh;
      chmod +x /setup_es_fixed.sh;
      /setup_es_fixed.sh --index-dir /es_index_schemas;
      touch /tmp/es_init_complete;
      echo 'Elasticsearch initialization completed successfully!'
    ) &
    exec /opt/bitnami/scripts/elasticsearch/entrypoint.sh /opt/bitnami/scripts/elasticsearch/run.sh;
    echo -e \"⏳ Adjusting Elasticsearch disk watermark settings...\"
  "

4. MinIO
docker run -d \
  --name coze-minio \
  --user root \
  --privileged \
  --restart always \
  -e MINIO_ROOT_USER=${MINIO_ROOT_USER:-minioadmin} \
  -e MINIO_ROOT_PASSWORD=${MINIO_ROOT_PASSWORD:-minioadmin123} \
  -e MINIO_DEFAULT_BUCKETS=${MINIO_BUCKET:-opencoze},${MINIO_DEFAULT_BUCKETS:-milvus} \
  -e STORAGE_BUCKET=opencoze \
  -p 9000:9000 \
  -p 9001:9001 \
  -v $PWD/data/minio:/data \
  -v $PWD/volumes/minio/default_icon/:/default_icon \
  -v $PWD/volumes/minio/official_plugin_icon/:/official_plugin_icon \
  --network coze-network \
  --health-cmd "/usr/bin/mc alias set health_check http://localhost:9000 ${MINIO_ROOT_USER} ${MINIO_ROOT_PASSWORD} && /usr/bin/mc ready health_check" \
  --health-interval 30s \
  --health-timeout 10s \
  --health-retries 3 \
  --health-start-period 60s \
  --entrypoint /bin/sh \
  minio/minio:RELEASE.2025-06-13T11-33-47Z-cpuv1 \
  -c '
    (
      until (/usr/bin/mc alias set localminio http://localhost:9000 $${MINIO_ROOT_USER} $${MINIO_ROOT_PASSWORD}) do
        echo "Waiting for MinIO to be ready..."
        sleep 1
      done
      /usr/bin/mc mb --ignore-existing localminio/$${STORAGE_BUCKET}
      /usr/bin/mc cp --recursive /default_icon/ localminio/$${STORAGE_BUCKET}/default_icon/
      /usr/bin/mc cp --recursive /official_plugin_icon/ localminio/$${STORAGE_BUCKET}/official_plugin_icon/
      echo "MinIO initialization complete."
    ) &
    exec minio server /data --console-address ":9001"
  '

5. etcd
docker run -d \
  --privileged \
  --name coze-etcd \
  --network coze-network \
  -p 2379:2379 \
  -p 2380:2380 \
  -v $PWD/data/bitnami/etcd:/bitnami/etcd:rw,Z \
  -v $PWD/volumes/etcd/etcd.conf.yml:/opt/bitnami/etcd/conf/etcd.conf.yml:ro,Z \
  -e ETCD_AUTO_COMPACTION_MODE=revision \
  -e ETCD_AUTO_COMPACTION_RETENTION=1000 \
  -e ETCD_QUOTA_BACKEND_BYTES=4294967296 \
  -e ALLOW_NONE_AUTHENTICATION=yes \
  --user root \
  --entrypoint /bin/bash \
  bitnami/etcd:3.5 \
  -c "
    /opt/bitnami/scripts/etcd/setup.sh
    chown -R etcd:etcd /bitnami/etcd
    chmod g+s /bitnami/etcd
    exec /opt/bitnami/scripts/etcd/entrypoint.sh /opt/bitnami/scripts/etcd/run.sh
  "

6. Milvus
docker run -d \
  --privileged=true \
  --name coze-milvus \
  --network coze-network \
  -p 19530:19530 \
  -p 9091:9091 \
  -v $PWD/data/milvus:/var/lib/milvus:rw,Z \
  -e ETCD_ENDPOINTS=coze-etcd:2379 \
  -e MINIO_ADDRESS=coze-minio:9000 \
  -e MINIO_BUCKET_NAME=${MINIO_BUCKET:-milvus} \
  -e MINIO_ACCESS_KEY_ID=${MINIO_ROOT_USER:-minioadmin} \
  -e MINIO_SECRET_ACCESS_KEY=${MINIO_ROOT_PASSWORD:-minioadmin123} \
  -e MINIO_USE_SSL=false \
  -e LOG_LEVEL=debug \
  --security-opt seccomp:unconfined \
  --entrypoint bash \
  milvusdb/milvus:v2.5.10 \
  -c "
    chown -R root:root /var/lib/milvus
    chmod g+s /var/lib/milvus
    exec milvus run standalone
  "

7. NSQ Lookupd
docker run -d \
  --name coze-nsqlookupd \
  --network coze-network \
  -p 4160:4160 \
  -p 4161:4161 \
  --restart always \
  --health-cmd "nsqlookupd --version" \
  --health-interval 5s \
  --health-timeout 10s \
  --health-retries 10 \
  --health-start-period 60s \
  nsqio/nsq:v1.2.1 \
  /nsqlookupd

8. NSQD
docker run -d \
  --name coze-nsqd \
  --network coze-network \
  -p 4150:4150 \
  -p 4151:4151 \
  --restart always \
  --health-cmd "nsqd --version" \
  --health-interval 5s \
  --health-timeout 10s \
  --health-retries 10 \
  --health-start-period 60s \
  --link nsqlookupd \
  nsqio/nsq:v1.2.1 \
  /nsqd --lookupd-tcp-address=coze-nsqlookupd:4160 --broadcast-address=nsqd

9. NSQAdmin
docker run -d \
  --name coze-nsqadmin \
  --network coze-network \
  --restart always \
  -p 4171:4171 \
  nsqio/nsq:v1.2.1 \
  /nsqadmin --lookupd-http-address=coze-nsqlookupd:4161

10. Coze Server
docker run -d \
  --name coze-server \
  --network coze-network \
  -p 8888:8888 \
  -p 8889:8889 \
  -v $PWD/coze_conf:/app/resources/conf \
  -e LISTEN_ADDR=0.0.0.0:8888 \
  --env-file .env.expanded \
  opencoze/opencoze:latest


2、如何注册或登录？
答：自己随便设置一个邮箱及密码，点击注册

3、如何查看日志？

答：docker logs coze-server

或者通过下面方法输出日志到文件

script -c "docker logs coze-server" coze-server.log

日志文件默认在：

/var/lib/docker/containers/<container-id>/<container-id>-json.log
(DEV)[root@zhangsan ~]# tail -f /var/lib/docker/containers/edcae5310cbc0e4567c297eef1b1b3ab788e4cbf4d6e680881ff715774da9154/edcae5310cbc0e4567c297eef1b1b3ab788e4cbf4d6e680881ff715774da9154-json.log

4、问答模型如何配置有什么要求？
答：问答模型需要满足如下输入输出要求，

从默认的配置文件中，复制一个yaml文件到model目录下，如：cp conf/model/template/model_template_openai.yaml  conf/model/open.yaml

修改yaml文件配置，主要是修改id,name,涉及到模型名称，及base_url,而且conn_config项下的base_url填写“http://192.168.82.29:8827/v2”即可，不需要填写后面的`/chat/completions`

特别注意：id不能是一样的，若在不同的模型文件中配置了相同的 id，导致模型 id 冲突，切换模型的时候就切换不了。

# 非流式问答输入输出：
curl -XPOST http://192.168.82.29:8827/v2/chat/completions   -H "Content-Type: application/json"   -d '{
    "model": "Qwen2.5-0.5B-Instruct",
    "messages": [{"role": "user", "content": "中国首都在哪里？"}]
  }'
{"choices":[{"message":{"role":"assistant","content":"中国首都是北京。"},"finish_reason":"stop","index":0}]} 

# 流式问答输入输出：
(DEV)[ecsuser@zhangsan fineAi]$ curl -XPOST http://192.168.82.29:8827/v2/chat/completions  -N -H "Content-Type: application/json" -H "accept: text/event-stream"  -d  '{
    "model": "Qwen2.5-0.5B-Instruct",
    "messages": [{"role": "user", "content": "中国首都在哪里？"}]
  }'
data:{"choices": [{"delta": {"content": ""}, "finish_reason": null, "index": 0, "logprobs": null}], "object": "chat.completion.chunk", "usage": null, "created": 1715931028, "system_fingerprint": null, "model": "qwen-plus", "id": "chatcmpl-3bb05cf5cd819fbca5f0b8d67a025022"}
data:{"choices": [{"delta": {"content": "中国"}, "finish_reason": null, "index": 0, "logprobs": null}], "object": "chat.completion.chunk", "usage": null, "created": 1715931028, "system_fingerprint": null, "model": "qwen-plus", "id": "chatcmpl-3bb05cf5cd819fbca5f0b8d67a025022"}
data:{"choices": [{"delta": {"content": "首"}, "finish_reason": null, "index": 0, "logprobs": null}], "object": "chat.completion.chunk", "usage": null, "created": 1715931028, "system_fingerprint": null, "model": "qwen-plus", "id": "chatcmpl-3bb05cf5cd819fbca5f0b8d67a025022"}
data:{"choices": [{"delta": {"content": "都是"}, "finish_reason": null, "index": 0, "logprobs": null}], "object": "chat.completion.chunk", "usage": null, "created": 1715931028, "system_fingerprint": null, "model": "qwen-plus", "id": "chatcmpl-3bb05cf5cd819fbca5f0b8d67a025022"}
data:{"choices": [{"delta": {"content": "北京"}, "finish_reason": null, "index": 0, "logprobs": null}], "object": "chat.completion.chunk", "usage": null, "created": 1715931028, "system_fingerprint": null, "model": "qwen-plus", "id": "chatcmpl-3bb05cf5cd819fbca5f0b8d67a025022"}
data:{"choices": [{"delta": {"content": "。"}, "finish_reason": null, "index": 0, "logprobs": null}], "object": "chat.completion.chunk", "usage": null, "created": 1715931028, "system_fingerprint": null, "model": "qwen-plus", "id": "chatcmpl-3bb05cf5cd819fbca5f0b8d67a025022"}
data:{"choices": [{"delta": {"content": ""}, "finish_reason": null, "index": 0, "logprobs": null}], "object": "chat.completion.chunk", "usage": null, "created": 1715931028, "system_fingerprint": null, "model": "qwen-plus", "id": "chatcmpl-3bb05cf5cd819fbca5f0b8d67a025022"}
data:{"choices": [{"delta": {"content": ""}, "finish_reason": "stop", "index": 0, "logprobs": null}], "object": "chat.completion.chunk", "usage": null, "created": 1715931028, "system_fingerprint": null, "model": "qwen-plus", "id": "chatcmpl-3bb05cf5cd819fbca5f0b8d67a025022"}
5、向量模型如何配置，有何要求？

答：向量模型，主要用户知识库搜索，其配置在`.env`环境变量文件中配置，主要配置如下项目：

同样的OPENAI_EMBEDDING_BASE_URL不需要向量服务url后面的`/embeddings`,仅仅填写前面的即可。

EMBEDDING_TYPE=openai
EMBEDDING_MAX_BATCH_SIZE=50
# openai embedding
OPENAI_EMBEDDING_BASE_URL=http://192.168.82.29:7866/compatible-mode/v1
# (string) OpenAI base_url
OPENAI_EMBEDDING_MODEL=bge-small-zh-v1.5
# (string) OpenAI embedding model
OPENAI_EMBEDDING_API_KEY=emb-9sF!2Lm@qW7xZ423we23e
# (string) OpenAI api_key
OPENAI_EMBEDDING_BY_AZURE=false
# (bool) OpenAI by_azure
OPENAI_EMBEDDING_API_VERSION=v1
# OpenAI azure api version
OPENAI_EMBEDDING_DIMS=512
其中，向量服务的输入输出要求是：

curl --location 'http://192.168.82.29:7866/compatible-mode/v1/embeddings' \
--header "Authorization: Bearer emb-9sF!2Lm@qW7xZ423we23e" \
--header 'Content-Type: application/json' \
--data '{
    "model": "bge-small-zh-v1.5",
    "input": ["衣服的质量杠杠的，很漂亮，不枉我等了这么久啊，喜欢，以后还来这里买"],
    "encoding_format": "float"
}' 

{
 "object": "list",
 "data": [
   {
     "object": "embedding",
     "embedding": [0.1, 0.2, 0.3, ...],
     "index": 0
   }
 ],
 "model": "bge-small-zh-v1.5",
 "usage": {
   "prompt_tokens": 5,
   "total_tokens": 5
 }
}

6、在哪里查看官方文档？
答：在https://github.com/coze-dev/coze-studio/wiki/6.-API-%E5%8F%82%E8%80%83 


