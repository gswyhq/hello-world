
# 用 `Pipeline` 上传数据
假设我们在127.0.0.1:9380中有一个FATE Flow Service（默认为单机版），然后执行
(venv) [root@97a23f46786b fate]# pipeline init --ip 127.0.0.1 --port 9380
# 检查是否启动fate flow server
(venv) [root@97a23f46786b fate]# pipeline config check
Flow server status normal, Flow version: 1.9.0


上载数据
在开始建模任务之前，应上传要使用的数据。
通常，一方通常是一个包含多个节点的集群。
因此，当我们上传这些数据时，数据将被分配给这些节点。

```python
from pipeline.backend.pipeline import PipeLine
# 创建管道实例：

# 注意，只需要本地 party id。
pipeline_upload = PipeLine().set_initiator(role='guest', party_id=9999).set_roles(guest=9999)
# 定义数据存储分区
partition = 4

# 定义将在FATE作业配置中使用的表名和命名空间
dense_data_guest = {"name": "breast_hetero_guest", "namespace": f"experiment"}
dense_data_host = {"name": "breast_hetero_host", "namespace": f"experiment"}
tag_data = {"name": "breast_hetero_host", "namespace": f"experiment"}

# 现在，我们添加要上载的数据
import os

data_base = "/workspace/FATE/"
pipeline_upload.add_upload_data(file=os.path.join(data_base, "examples/data/breast_hetero_guest.csv"),
                                table_name=dense_data_guest["name"],             # table name
                                namespace=dense_data_guest["namespace"],         # 命名空间 namespace
                                head=1, partition=partition)               # data info

pipeline_upload.add_upload_data(file=os.path.join(data_base, "examples/data/breast_hetero_host.csv"),
                                table_name=dense_data_host["name"],
                                namespace=dense_data_host["namespace"],
                                head=1, partition=partition)

pipeline_upload.add_upload_data(file=os.path.join(data_base, "examples/data/breast_hetero_host.csv"),
                                table_name=tag_data["name"],
                                namespace=tag_data["namespace"],
                                head=1, partition=partition)

# 然后，我们可以上传数据
pipeline_upload.upload(drop=1)

# 单机版 FATE 最终保路径：
# [root@97a23f46786b experiment]# pwd
# /data/projects/fate/data/experiment
# [root@97a23f46786b experiment]# ls
# breast_hetero_guest  breast_hetero_host  breast_homo_guest  breast_homo_host
# experiment：为对应的命名空间
# breast_hetero_guest  breast_hetero_host  breast_homo_guest  breast_homo_host: 为对应的表名
# [root@97a23f46786b experiment]# cd breast_hetero_host
# [root@97a23f46786b breast_hetero_host]# ls
# 0  1  2  3
# [root@97a23f46786b breast_hetero_host]# cd 0
# [root@97a23f46786b 0]# ls
# data.mdb  lock.mdb

# 上传完成后，会有这样的日志信息：
# 2022-10-13 07:00:31.643 | INFO     | pipeline.utils.invoker.job_submitter:monitor_job_status:89 - Job is success!!! Job id is 202210130700241548080
# 浏览器打开，对应的服务，如 192.168.3.101:8080
# 点击右上角“JOBS”, 按 job id 找到对应的任务；点击，进入“Job detail”页面；
# 点击“Main Graph”上的节点，即可在右侧看到该任务id对应的信息；

```
1、浏览器打开，对应的服务，如 192.168.3.101:8080
2、点击右上角“JOBS”, 按 job id 找到对应的任务，注意host、guest角色(不同角色看到的结果是不一样的)；点击，进入“Job detail”页面；
3、点击“Main Graph”上的节点，如选中节点“reader 1”，再点击右下角的“view the outputs”，即可以看到对应读取的数据；


# 数据集说明：
examples/data/breast_hetero_guest.csv 和 examples/data/breast_hetero_host.csv，
包含标题行在内，均是570行；
其中examples/data/breast_hetero_guest.csv 的标题行为：
id,y,x0,x1,x2,x3,x4,x5,x6,x7,x8,x9
examples/data/breast_hetero_host.csv 的标题行为：
id,x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19
并且两个文件的id列，最小id均为0，最大id均为568，共569个不重复的id；
数据集使用sklearn库内置的乳腺癌肿瘤数据集，为了模拟横向联邦学习，将数据集切分为特征相同的横向联邦形式。
该数据集一共有569条数据，30个特征数（其实是10个属性，分别以均值mean、标准差std、最差值worst出现了三次），
一个标签（1：良性肿瘤、0：恶行肿瘤，1：0=357：212）。

