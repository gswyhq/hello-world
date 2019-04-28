
第一步：重写Dockerfile 文件
在文件头添加两行
```
FROM openjdk:8-jre-alpine
RUN sed -i '1iwebot:x:0:0:webot:/webot:/bin/ash' /etc/passwd && su webot
USER webot
...
```

第二步：通过Dockerfile生成neo4j镜像
`docker build -t my_neo4j:v1 -f Dockerfile .`

第三步：指定用户启动neo4j容器
`docker run --rm -it --publish=7474:7474 --publish=7687:7687 -u webot --env=NEO4J_AUTH=neo4j/gswyhq  my_neo4j:v1 `
