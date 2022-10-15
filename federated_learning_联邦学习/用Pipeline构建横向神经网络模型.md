
# 第一步
连接 FATE Flow  服务
!pipeline init --ip 127.0.0.1 --port 9380

# 上传数据
```python
from pipeline.backend.pipeline import PipeLine
# 创建管道实例：

# 注意，只需要本地 party id。
pipeline_upload = PipeLine().set_initiator(role='guest', party_id=9999).set_roles(guest=9999)
# 定义数据存储分区
partition = 4

# 定义将在FATE作业配置中使用的表名和命名空间
dense_data_guest = {"name": "breast_homo_guest", "namespace": f"experiment"}
dense_data_host = {"name": "breast_homo_host", "namespace": f"experiment"}

# 现在，我们添加要上载的数据
import os

data_base = "."
pipeline_upload.add_upload_data(file=os.path.join(data_base, "./examples/data/breast_homo_guest.csv"),
                                table_name=dense_data_guest["name"],             # table name
                                namespace=dense_data_guest["namespace"],         # 命名空间 namespace
                                head=1, partition=partition)               # data info

pipeline_upload.add_upload_data(file=os.path.join(data_base, "./examples/data/breast_homo_host.csv"),
                                table_name=dense_data_host["name"],
                                namespace=dense_data_host["namespace"],
                                head=1, partition=partition)

# 然后，我们可以上传数据
pipeline_upload.upload(drop=1)


```

```python
from pipeline.backend.pipeline import PipeLine
from pipeline.component import DataTransform
from pipeline.component import Reader
from pipeline.component import HomoNN
from pipeline.interface import Data

# 创建 pipeline 实例:

# - initiator: 
#     * role: guest
#     * party: 9999
# - roles:
#     * guest: 9999
#     * host: [10000, 9999]
#     * arbiter: 9999

pipeline = PipeLine() \
        .set_initiator(role='guest', party_id=9999) \
        .set_roles(guest=9999, host=[10000], arbiter=10000)

reader_0 = Reader(name="reader_0")
# set guest parameter
reader_0.get_party_instance(role='guest', party_id=9999).component_param(
    table={"name": "breast_homo_guest", "namespace": "experiment"})
# set host parameter
reader_0.get_party_instance(role='host', party_id=10000).component_param(
    table={"name": "breast_homo_host", "namespace": "experiment"})


data_transform_0 = DataTransform(name="data_transform_0", with_label=True)
# set guest parameter
data_transform_0.get_party_instance(role='guest', party_id=9999).component_param(
    with_label=True)
data_transform_0.get_party_instance(role='host', party_id=[10000]).component_param(
    with_label=True)

homo_nn_0 = HomoNN(
    name="homo_nn_0", 
    max_iter=10, 
    batch_size=-1, 
    early_stop={"early_stop": "diff", "eps": 0.0001})

from tensorflow.keras.layers import Dense
homo_nn_0.add(
    Dense(units=1, input_shape=(10,), activation="sigmoid"))

from tensorflow.keras import optimizers
homo_nn_0.compile(
    optimizer=optimizers.Adam(learning_rate=0.05), 
    metrics=["accuracy", "AUC"],
    loss="binary_crossentropy")

pipeline.add_component(reader_0)
pipeline.add_component(data_transform_0, data=Data(data=reader_0.output.data))
pipeline.add_component(homo_nn_0, data=Data(train_data=data_transform_0.output.data))
pipeline.compile();

pipeline.fit()

summary = pipeline.get_component("homo_nn_0").get_summary()
summary
# {'is_converged': False, 'loss_history': [0.7287938594818115, 0.4937364459037781, 0.3601800501346588, 0.27961793541908264, 0.22668775916099548, 0.1895807981491089, 0.16256916522979736, 0.14249403774738312, 0.12731970846652985, 0.11564338207244873]}

pylab.plot(summary['loss_history'])

```