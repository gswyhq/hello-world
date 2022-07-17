
# 启动服务：
python -m sql_app.main

启动服务后，会在项目目录下生成一个sqlite数据文件：sql_app.db ；并包含两个空数据表：items、users
需要向 sql_app.db 添加了数据，才可以查询，否则查询不到结果。
INSERT INTO users (id,email,hashed_password,is_active) VALUES (1,'zs@126.com','52341',1);

# 查询数据：
curl -X 'GET'   'http://localhost:8000/users/1'   -H 'accept: application/json'
{"email":"zs@126.com","id":1,"is_active":true,"items":[]} 


