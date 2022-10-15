
# 连接 FATE Flow Service
假设我们在127.0.0.1:9380中有一个FATE Flow Service（默认为单机版），然后执行
(venv) [root@97a23f46786b fate]# pipeline init --ip 127.0.0.1 --port 9380
# 检查是否启动fate flow server
(venv) [root@97a23f46786b fate]# pipeline config check
Flow server status normal, Flow version: 1.9.0

```python
from pipeline.backend.pipeline import PipeLine
from pipeline.component import Reader, DataTransform, Intersection, HeteroSecureBoost, Evaluation
from pipeline.interface import Data

# 创建一个 pipeline 实例:
# - initiator: 
#     * role: guest
#     * party: 9999
# - roles:
#     * guest: 9999
#     * host: 10000
    
pipeline = PipeLine() \
        .set_initiator(role='guest', party_id=9999) \
        .set_roles(guest=9999, host=10000)

# 定义一个 Reader 加载数据
reader_0 = Reader(name="reader_0")
# set guest parameter
reader_0.get_party_instance(role='guest', party_id=9999).component_param(
    table={"name": "breast_hetero_guest", "namespace": "experiment"})
# set host parameter
reader_0.get_party_instance(role='host', party_id=10000).component_param(
    table={"name": "breast_hetero_host", "namespace": "experiment"})

# 添加DataTransform组件以将原始数据解析到数据实例中
data_transform_0 = DataTransform(name="data_transform_0")
# set guest parameter
data_transform_0.get_party_instance(role='guest', party_id=9999).component_param(
    with_label=True)
data_transform_0.get_party_instance(role='host', party_id=[10000]).component_param(
    with_label=False)

# 添加Intersection组件以执行 hetero-scenario 的PSI
intersect_0 = Intersection(name="intersect_0")

# 现在，我们定义了HeteroSecureBoost组件。将为所有相关方设置以下参数。
hetero_secureboost_0 = HeteroSecureBoost(name="hetero_secureboost_0",
                                         num_trees=5,
                                         bin_num=16,
                                         task_type="classification",
                                         objective_param={"objective": "cross_entropy"},
                                         encrypt_param={"method": "paillier"},
                                         tree_param={"max_depth": 3})
# 为了显示评估结果，需要一个“评估”组件。
evaluation_0 = Evaluation(name="evaluation_0", eval_type="binary")

# 按执行顺序将组件添加到管道
# -data_transform0 comsume reader0的输出数据
# -intersect_0 comsume data_transform0的输出数据
# -hetro_secureboost0使用intersect_0的输出数据
# -evaluation0在训练数据上使用异质安全成本0的预测结果

pipeline.add_component(reader_0)
pipeline.add_component(data_transform_0, data=Data(data=reader_0.output.data))
pipeline.add_component(intersect_0, data=Data(data=data_transform_0.output.data))
pipeline.add_component(hetero_secureboost_0, data=Data(train_data=intersect_0.output.data))
pipeline.add_component(evaluation_0, data=Data(data=hetero_secureboost_0.output.data))

# 然后编译我们的管道，以便准备提交。
pipeline.compile();

# 训练模型：
pipeline.fit()

# 一旦训练完成，训练好的模型就可以用于预测。或者，保存经过训练的管道以备将来使用。
pipeline.dump("pipeline_saved_20221013_1640.pkl");
# 返回结果是反序列化数据，如： \x80\x04\x95，80是协议的意思，04则表示为这是4版本，95则类似一个框架

# 首先，从管道部署所需组件：
pipeline = PipeLine.load_model_from_file('pipeline_saved_20221013_1640.pkl')
pipeline.deploy_component([pipeline.data_transform_0, pipeline.intersect_0, pipeline.hetero_secureboost_0]);

# 定义用于读取预测数据的新读卡器组件
reader_1 = Reader(name="reader_1")
reader_1.get_party_instance(role="guest", party_id=9999).component_param(table={"name": "breast_hetero_guest", "namespace": "experiment"})
reader_1.get_party_instance(role="host", party_id=10000).component_param(table={"name": "breast_hetero_host", "namespace": "experiment"})

# （可选）定义新的评估组件。
evaluation_0 = Evaluation(name="evaluation_0", eval_type="binary")

# 按执行顺序添加组件以预测管道：
predict_pipeline = PipeLine()
predict_pipeline.add_component(reader_1)\
                .add_component(pipeline, 
                               data=Data(predict_input={pipeline.data_transform_0.input.data: reader_1.output.data}))\
                .add_component(evaluation_0, data=Data(data=pipeline.hetero_secureboost_0.output.data));

# 开始预测
predict_pipeline.predict()

```
# 查看模型预测结果
模型预测完成后，会有类似这样的：
 Job is success!!! Job id is 202210130914535702300
1、浏览器打开，对应的服务，如 192.168.3.101:8080
2、点击右上角“JOBS”, 按 job id 找到对应的任务，注意host、guest角色(不同角色看到的结果是不一样的)；点击，进入“Job detail”页面；
3、点击“Main Graph”上的节点，如选中节点“reader 1”，再点击右下角的“view the outputs”，即可以看到对应读取的数据；
4、选中节点“hetero_secureboost_0”,再点击右下角的“view the outputs”，即可以看到模型评估详情结果；
5、选中节点“evaluation_0”，再点击右下角的“view the outputs”，即可以看到模型评估的各项指标。
