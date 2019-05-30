
chinese-bert-service

docker build -t gswyhq/chinese-bert-service --no-cache -f Dockerfile .

docker run --rm -it -p 15555:5555 -p 15556:5556 gswyhq/chinese-bert-service bert-serving-start -num_worker=1 -model_dir /tmp/chinese_L-12_H-768_A-12

```python
from bert_serving.client import BertClient

bc = BertClient(ip='192.168.3.164', port=15555, port_out=15556)
bc.encode(['我喜欢你们','我喜欢你们','我喜欢你'])
Out[14]: 
array([[ 0.39121535, -0.14233316,  0.04368735, ..., -0.19955774,
         0.6797884 , -0.58785284],
       [ 0.39121535, -0.14233316,  0.04368735, ..., -0.19955774,
         0.6797884 , -0.58785284],
       [ 0.33524945,  0.03421693, -0.04716328, ..., -0.19211133,
         0.37312028, -0.71963006]], dtype=float32)

```

[reference](https://github.com/hanxiao/bert-as-service.git)

