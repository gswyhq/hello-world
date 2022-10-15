
资料来源：https://github.com/FederatedAI/FATE

一、下载docker镜像（单机版）：
wget -c -t 0 https://webank-ai-1251170195.cos.ap-guangzhou.myqcloud.com/fate/1.9.0/release/standalone_fate_docker_image_1.9.0_release.tar.gz

docker load -i standalone_fate_docker_image_${version}_release.tar.gz
docker images | grep federatedai/standalone_fate
能看到对应${version}的镜像则镜像下载成功

集群部署可以参考：https://github.com/FederatedAI/FATE/blob/v1.8.0/deploy/cluster-deploy/doc/fate_on_eggroll/fate-allinone_deployment_guide.zh.md

2.3 启动
检查本地8080端口是否被占用
netstat -apln|grep 8080

docker run -d --name standalone_fate -p 8080:8080 federatedai/standalone_fate:${version};
docker ps -a | grep standalone_fate
能看到对应${version}的容器运行中则启动成功

2.4 测试
进入容器
docker exec -it $(docker ps -aqf "name=standalone_fate") bash
# 加载环境变量
source bin/init_env.sh

4. 测试项
4.1 Toy测试
flow test toy -gid 10000 -hid 10000
如果成功，屏幕显示类似下方的语句:
success to calculate secure_sum, it is 2000.0

4.2 单元测试
fate_test unittest federatedml --yes
如果成功，屏幕显示类似下方的语句:

there are 0 failed test


有些用例算法在 examples 文件夹下, 请尝试使用。

您还可以通过浏览器体验算法过程看板，访问：Http://${ip}:8080, ip为127.0.0.1或本机实际ip
默认用户名/密码：admin/admin

3.Fate部署架构说明
3.1组件说明
软件产品    组件  端口  说明
fate    fate_flow   9360;9380   联合学习任务流水线管理模块
fate    fateboard   8080    联合学习过程可视化模块
fate    FederatedML -    算法代码包
eggroll clustermanager  4670    cluster manager管理集群
eggroll nodemanger  4671    node manager管理每台机器资源
eggroll rollsite    9370    跨站点或者跨party通讯组件
mysql   mysql   3306    数据存储，clustermanager和fateflow依赖


