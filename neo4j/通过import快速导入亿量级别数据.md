
`neo4j-admin import`可实现快速导入大量数据到neo4j;但需要一个前提条件是数据库数据是空的；
比如，若导入默认数据库`/var/lib/neo4j/data/databases/graph.db`，则需要在导入之前将该目录删除掉；
意味着该方法不能在已有数据的neo4j数据库重大导入数据；也不能实现数据的热导入。

可以支持 zip 或 gzip 进行的压缩文件，但每个压缩文件必须包含且仅包含一个文件。

### 准备数据文件
1. 数据示例：

`movies-header.csv`
```shell
movieId:ID,title,year:int,:LABEL
```

`movies.csv`
```shell
tt0133093,"The Matrix",1999,Movie
tt0234215,"The Matrix Reloaded",2003,Movie;Sequel
tt0242653,"The Matrix Revolutions",2003,Movie;Sequel
```

`actors-header.csv`

```shell
personId:ID,name,:LABEL
```

`actors.csv`

```shell
keanu,"Keanu Reeves",Actor
laurence,"Laurence Fishburne",Actor
carrieanne,"Carrie-Anne Moss",Actor
```

`roles-header.csv`
```shell
:START_ID,role,:END_ID,:TYPE
```

`roles.csv`
```shell
keanu,"Neo",tt0133093,ACTED_IN
keanu,"Neo",tt0234215,ACTED_IN
keanu,"Neo",tt0242653,ACTED_IN
laurence,"Morpheus",tt0133093,ACTED_IN
laurence,"Morpheus",tt0234215,ACTED_IN
laurence,"Morpheus",tt0242653,ACTED_IN
carrieanne,"Trinity",tt0133093,ACTED_IN
carrieanne,"Trinity",tt0234215,ACTED_IN
carrieanne,"Trinity",tt0242653,ACTED_IN
```

### 对数据文件进行压缩（当然也可以不压缩）
```shell
zip actors.csv.zip actors.csv
gzip -c movies.csv > movies.csv.gz
gzip -c roles.csv > roles.csv.gz

```

### 将数据导入到neo4j

1. 本数据在`Neo4j Community version: 3.5.6; 3.5.4; 3.4.6; 3.4.5; 3.2.5`测试通过；
2. 但有些数据在`Neo4j Community version: 3.2.5`上测试不通过；

neo4j_home$ ls import
actors-header.csv  actors.csv.zip  movies-header.csv  movies.csv.gz  roles-header.csv  roles.csv.gz
neo4j_home$ bin/neo4j-admin import --nodes import/movies-header.csv,import/movies.csv.gz --nodes import/actors-header.csv,import/actors.csv.zip --relationships import/roles-header.csv,import/roles.csv.gz


### 参考资料
https://neo4j.com/docs/operations-manual/3.5/tools/import/syntax/

https://neo4j.com/docs/operations-manual/3.5/tools/import/file-header-format/#import-tool-header-format-nodes

转义字符格式：
如： o13713,"6'1\" (185 cm)",Tongyonggraph;Tongyong
应该写成： o13713,"6'1"" (185 cm)",Tongyonggraph;Tongyong
也就是说字符串中若有双引号，应该替换的两个连续的双引号
