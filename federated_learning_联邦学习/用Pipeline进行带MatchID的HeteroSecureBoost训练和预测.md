
管道匹配ID教程
从版本1.7开始，FATE区分样本id（sid）和匹配id。sid对于每个样本条目是唯一的，而匹配id对应于单个样本源标识。
此自适应允许FATE对具有重复匹配id的样本执行私有集交集。用户可以选择在上载时将uuid附加到原始样本条目来创建sid；
然后，模块DataTransform将提取真正的匹配id以供以后使用。

# 上传数据
```python
import os 
from pipeline.backend.pipeline import PipeLine

pipeline_upload = PipeLine().set_initiator(role='guest', party_id=9999).set_roles(guest=9999)
partition = 4
dense_data_guest = {"name": "breast_hetero_guest", "namespace": f"experiment"}
dense_data_host = {"name": "breast_hetero_host", "namespace": f"experiment"}

# 现在，要将uuid创建为sid，请启用extend_sid选项。或者，设置auto_increasing_sid使扩展sid从0开始。
data_base = "."
pipeline_upload.add_upload_data(file=os.path.join(data_base, "examples/data/breast_hetero_guest.csv"),
                                table_name=dense_data_guest["name"],             # table name
                                namespace=dense_data_guest["namespace"],         # namespace
                                head=1, partition=partition,                     # data info
                                extend_sid=True,                                 # extend sid 
                                auto_increasing_sid=False)

pipeline_upload.add_upload_data(file=os.path.join(data_base, "examples/data/breast_hetero_host.csv"),
                                table_name=dense_data_host["name"],
                                namespace=dense_data_host["namespace"],
                                head=1, partition=partition,
                                extend_sid=True,
                                auto_increasing_sid=False) 

pipeline_upload.upload(drop=1)

```


```python
from pipeline.backend.pipeline import PipeLine
from pipeline.component import Reader, DataTransform, Intersection, HeteroSecureBoost, Evaluation
from pipeline.interface import Data

pipeline = PipeLine() \
        .set_initiator(role='guest', party_id=9999) \
        .set_roles(guest=9999, host=10000)

reader_0 = Reader(name="reader_0")
# set guest parameter
reader_0.get_party_instance(role='guest', party_id=9999).component_param(
    table={"name": "breast_hetero_guest", "namespace": "experiment"})
# set host parameter
reader_0.get_party_instance(role='host', party_id=10000).component_param(
    table={"name": "breast_hetero_host", "namespace": "experiment"})

#  set with match id
data_transform_0 = DataTransform(name="data_transform_0", with_match_id=True)
# set guest parameter
data_transform_0.get_party_instance(role='guest', party_id=9999).component_param(
    with_label=True)
data_transform_0.get_party_instance(role='host', party_id=[10000]).component_param(
    with_label=False)

intersect_0 = Intersection(name="intersect_0")

hetero_secureboost_0 = HeteroSecureBoost(name="hetero_secureboost_0",
                                         num_trees=5,
                                         bin_num=16,
                                         task_type="classification",
                                         objective_param={"objective": "cross_entropy"},
                                         encrypt_param={"method": "paillier"},
                                         tree_param={"max_depth": 3})

evaluation_0 = Evaluation(name="evaluation_0", eval_type="binary")


pipeline.add_component(reader_0)
pipeline.add_component(data_transform_0, data=Data(data=reader_0.output.data))
pipeline.add_component(intersect_0, data=Data(data=data_transform_0.output.data))
pipeline.add_component(hetero_secureboost_0, data=Data(train_data=intersect_0.output.data))
pipeline.add_component(evaluation_0, data=Data(data=hetero_secureboost_0.output.data))
pipeline.compile();

pipeline.fit()
pipeline.get_component("data_transform_0").get_output_data(limits=3)
#                           extend_sid   id  y         x0         x1         x2         x3         x4         x5         x6         x7         x8         x9
# 0  479a2cbe4b5d11ed90d50242ac1100090  133  1   0.254879  -1.046633   0.209656   0.074214  -0.441366  -0.377645  -0.485934   0.347072   -0.28757  -0.733474
# 1  479a2cbe4b5d11ed90d50242ac1100091  273  1  -1.142928  -0.781198  -1.166747  -0.923578    0.62823  -1.021418  -1.111867  -0.959523  -0.096672  -0.121683

```
