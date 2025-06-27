
# 问题：模型蒸馏时，出现错误：
element 0 of tensors does not require grad and does not have a grad_fn
问题原因及解决方法：
出现这种问题是因为我们需要对变量求梯度，但是系统默认的是False, 也就是不对这个变量求梯度。
loss.backward()之前添加：
loss = Variable(loss, requires_grad = True)

# torch.load(model_path)出错：
保存模型：torch.save(model, 'model.pkl')
但是加载模型：
model = torch.load('model.pkl')
报错：
ModuleNotFoundError: No module named '***'
解决方案，修改保存模型的方法：
在保存模型同样的环境下加载。

# 错误：
AttributeError: 'Tensor' object has no attribute 'numpy'
原因可能有两个
第一个是TensorFlow的版本问题，要TensorFlow1.14以上版本才有，所以就解决方案就是升级TensorFlow到1.14以上版本
具体语句为pip install tensorflow==版本号
或者改写代码：
y = result.numpy()
改为：
with tf.Session() as sess:
    y = sess.run(result)

第二个原因，如果你升级了以后还是报错那么就添加以下语句tf.enable_eager_execution()
切换到eager模式即可解决。

解决方式：（版本1.14后才有的特性）
import tensorflow as tf
tf.enable_eager_execution() # 关键
 
m = tf.keras.metrics.Accuracy()
m.update_state([1, 2, 3, 4], [1, 2, 3, 4])
print('Final result: ', m.result().numpy())

# 错误：
RuntimeError: tf.placeholder() is not compatible with eager execution.
解决方法：
在代码中添加这样一句：
tf.compat.v1.disable_eager_execution()
如：
import tensorflow as tf
import numpy as np
tf.compat.v1.disable_eager_execution()

# 模型在预测时，开始时候正常，预测到后面报错，如预测到10%出现错误：
RuntimeError: CUDA out of memory. Tried to allocate 12.00MiB (GPU 0; 15.75 GiB total capacity; 14.25 GiB already allocated; 1.62 MiB free; 368.20 MiB cached)
问题原因及解决方法：
主要是在预测时候没有禁止求梯度，改为下面这样就可以了：
with torch.no_grad():
    logits = model(input_ids, input_mask, None)

# Pytorch: list, numpy. Tensor 格式转化 （附 only one element tensors can be converted to Python scalars 解决）
# list -> torch.Tensor 转 numpy
a = torch.tensor([1,2,3])
a
Out[9]: tensor([1, 2, 3])
a.numpy()
Out[10]: array([1, 2, 3], dtype=int64)

# numpy 转 torch.Tensor
torch.from_numpy(np.array([1,2,3]))
Out[11]: tensor([1, 2, 3], dtype=torch.int32)

# list 转 torch.Tensor
torch.tensor([1,2,3])
Out[12]: tensor([1, 2, 3])

注意：有时，上面操作会出现报错：ValueError:only one element tensors can be converted to Python scalars
原因是：要转换的list里面的元素包含多维的tensor。
在 gpu 上的解决方法是：
val= torch.tensor([item.cpu().detach().numpy() for item in val]).cuda() 
这是因为 gpu上的 tensor 不能直接转为 numpy； 需要先在 cpu 上完成操作，再回到 gpu 上
如果是在 cpu 上，上面的 .cpu() 和 .cuda() 可以省略

# torch.Tensor 转 list
list = tensor.numpy().tolist()   # 先转 numpy，后转 list
ts = torch.tensor([1,2,3])
ts
Out[17]: tensor([1, 2, 3])
ts.numpy().tolist()
Out[18]: [1, 2, 3]

# list 转 numpy
ndarray = np.array(list)

# numpy 转 list
list = ndarray.tolist()

# import torch 导入报错：
OSError: [WinError 126] 找不到指定的模块。 Error loading "D:\Users\Users1\AppData\Roaming\Python\Python36\site-packages\torch\lib\asmjit.dll" or one of its dependencies.
可能是因为安装的torch版本不对，重新安装对应的torch版本：
可在下面的页面查找到对应的torch版本：
https://download.pytorch.org/whl/torch_stable.html
根据前面的对应关系，下载好适合你的版本的 torch 、torchvision。
cu102 # 表示CUDA=10.2
cp37 # 表示python=3.7
linux or win # 表示对应的宿主机操作系统环境
下载好后，用pip安装，先cd 到下载的文件夹
pip install torch-1.7.0+cu101-cp36-cp36m-win_amd64.whl
pip install torchvision-0.8.0-cp36-cp36m-win_amd64.whl
测试GPU版本的torch是否安装成功
(torch) D:\MyData\xiaCN\Desktop\Work\unbiased> python
Python 3.6.13 (default, Feb 19 2021, 05:17:09) [MSC v.1916 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license" for more information.
>>> import torch
>>> torch.cuda.is_available()
True

# 但有时候版本对应好了，但是import troch还是报错：
>>> import torch
Microsoft Visual C++ Redistributable is not installed, this may lead to the DLL load failure.
                 It can be downloaded at https://aka.ms/vs/16/release/vc_redist.x64.exe
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "D:\Users\user3\AppData\Roaming\Python\Python36\site-packages\torch\__init__.py", line 135, in <modu
le>
    raise err
OSError: [WinError 126] 找不到指定的模块。 Error loading "D:\Users\user3\AppData\Roaming\Python\Python36\si
te-packages\torch\lib\asmjit.dll" or one of its dependencies.
# 实际上错误原因是说了，是因为 没有安装 Microsoft Visual C++ Redistributable, 在 https://aka.ms/vs/16/release/vc_redist.x64.exe 下载安装即可。
# 但本人发现自己电脑安装的Microsoft Visual C++ Redistributable，却为何还报这个错误呢，猜测可能是因为Microsoft Visual C++ Redistributable版本不对。
于是卸载本机已经安装的 Microsoft Visual C++ 2010 Redistributable
重新安装 Microsoft Visual C++ 2015-2019 Redistributable，安装完成后，重启电脑，结果就正常了：
~$ python
Python 3.6.5 |Anaconda, Inc.| (default, Mar 29 2018, 13:32:41) [MSC v.1900 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license" for more information.
>>> import torch
>>> torch.__version__
'1.9.0+cpu'
>>> torch.cuda.is_available()
False

# 模型训练时候，最后一轮batch_size 不匹配：
ValueError: Expected input batch_size (32) to match target batch_size (7).
除了模型结构存在问题外，也有可能是定义字段时候，batch_first 没有设置为True;
batch_first默认值为False，batch_first:如果为假，则输入和输出张量以(seq_len, batch)的形式提供。
batch_first:如果为真，则输入和输出张量以(batch, seq_len)的形式提供。
若以以(seq_len, batch)的形式提供，会与target的batch_size (batch)不匹配；
TEXT = data.Field(sequential=True, tokenize=x_tokenize, fix_length=BATCH_SIZE, include_lengths=True, use_vocab=True, batch_first=True)

# pytorch 读取自带数据集，速度慢的问题：
读取自带数据集，速度慢，可以预先下载好数据集，再从指定路径读取即可：
如：
from torchvision import datasets
from torchvision import transforms
train_dataset = datasets.MNIST(root='~/.torch', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='~/.torch', train=False, transform=transforms.ToTensor(), download=True)
预先下载好下面四个文件：
https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz
https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz
https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz
https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz
创建层级目录：mkdir -p ~/.torch/MNIST/raw
并将 下载好的四个文件放到：~/.torch/MNIST/raw 路径下（如下所示）；datasets.MNIST类初始化参数root路径填写：'~/.torch'，当然也可以是其他的，但届时，MNIST/raw目录也需要移至到相应目录下即可；
~/.torch tree
.
+--- MNIST
|   +--- raw
|   |   +--- t10k-images-idx3-ubyte.gz
|   |   +--- t10k-labels-idx1-ubyte.gz
|   |   +--- train-images-idx3-ubyte.gz
|   |   +--- train-labels-idx1-ubyte.gz

# 训练模型出错：
RuntimeError: Expected tensor for argument #1 'indices' to have one of the following scalar types: Long, Int; but got torch.FloatTensor instead (while checking arguments for embedding)
问题原因及解决方法：输入参数需要转成long类型才能作为nn.embedding层的输入；
inputs = inputs.long()

# 有时候数据类型转换警告：
UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
在使用torch.tensor()， 对某个变量进行转换的时候会报这个错误。我们只需要将其换成torch.as_tensor()就好了。
原来的代码：
x = torch.tensor(x)
修改为：
x = torch.as_tensor(x)

# 训练时候报错：
IndexError: Dimension out of range (expected to be in range of [-1, 0], but got 1)
模型预测值V在训练阶段最后一个step时，tensor维度从二维变成一维，导致出错。
例：训练集个数81，bs设置为4时，最后一个step 只剩1张图片。最后step时tensor维度从二维变成一维。
于是添加以下判断语句解决问题。如果有其他方法，可以交流一下
 v = net(inputs)
 if len(v.shape)==1:  #方式出现训练最后一个step时，出现v是一维的情况
     v=torch.unsqueeze(v,0)  
 loss = loss_func(v, targets)

# 问题：torch.jit.trace(model, (inputs_a['input_ids'], inputs_a['attention_mask'], inputs_a['token_type_ids']) )转换模型出错：
RuntimeError: Encountering a dict at the output of the tracer might cause the trace to be incorrect, this is only valid if the container structure does not change based on the module's inputs. Consider using a constant container instead (e.g. for `list`, use a `tuple` instead. for `dict`, use a `NamedTuple` instead). If you absolutely need this and know the side effects, pass strict=False to trace() to allow this behavior.
可能错误原因
网络输出为list或dict出现错误
解决方案
将输出用tuple和NamedTuple包裹。
如：
from collections import namedtuple
from typing import Any
import torch
# pylint: disable = abstract-method
class ModelWrapper(torch.nn.Module):
    """
    Wrapper class for model with dict/list rvalues.
    """
    def __init__(self, model: torch.nn.Module) -> None:
        """
        Init call.
        """
        super().__init__()
        self.model = model
    def forward(self, input_x: torch.Tensor, *args, **kwargs) -> Any:
        """
        Wrap forward call.
        """
        data = self.model(input_x, *args, **kwargs)
        if isinstance(data, dict):
            data_named_tuple = namedtuple("ModelEndpoints", sorted(data.keys()))  # type: ignore
            data = data_named_tuple(**data)  # type: ignore
        elif isinstance(data, list):
            data = tuple(data)
        return data

script_model = torch.jit.trace(  ModelWrapper(model), (inputs_a['input_ids'], inputs_a['attention_mask'], inputs_a['token_type_ids'],))
script_model.save('script_model.pt')
~$ netron script_model.pt

# 出现异常：
import torchmetrics
出现：
OSError: libcudart.so.10.2: cannot open shared object file: No such file or directory
可能是因为对应版本与cuda版本不匹配，降低或升高对应版本尝试即可；
如：因使用的是：torch 1.12.0+cu113， 
pip3 install -U torchmetrics==0.9.0即可解决问题；


# torch.load加载模型报错：
ModuleNotFoundError: No module named 'det'
用torch.save(model, checkpoint_path)保存的模型文件,那么会把对应的网络结构路径序列化到模型内部, 而一旦更改了网络结构(model/slim)路径,再torch.load()时会报错
主要是因为保存与加载模型不在同一个环境，导致自定义参数无法识别；
解决方法，保存模型的时候，网络结构、模型权重参数、优化器参数都保存
custom_model = {'net': CNN(),
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
                }

torch.save(custom_model, 'custom_model.pkl')
# 保存后的文件使用torch.load()后可以通过字典取值方式获取net、model_state_dict等键值内容。
custom_model = torch.load('custom_model.pkl')
model = custom_model['net']
model.load_state_dict(custom_model['model_state_dict'])

或者在load之前将原有模型保存路径添加到环境变量中：
如：sys.path.append(rf"D:\Users\{USERNAME}\github_project/paddle2torch_PPOCRv3")
之后再torch.load(...

# 模型预测时候报错：
RuntimeError: Expected q_dtype == at::kHalf || q_dtype == at::kBFloat16 to be true, but got false.  (Could this error message be improved?  If so, please report an enhancement request to PyTorch.)
解决方法：
第一步，确认设备是否支持bfloat16:
import transformers
transformers.utils.import_utils.is_torch_bf16_gpu_available()
若为真，则支持；
第二步，模型转换：
        # 将状态字典中的权重转换为bfloat16
        for key in state_dict:
            state_dict[key] = state_dict[key].bfloat16()

        # 加载到模型中
        model.load_state_dict(state_dict, strict=False)

        # 设置模型为bfloat16浮点运算模式（如果模型支持）
        model.to(torch.bfloat16)

# 问题，为何cpu环境pip安装torch时候，会安装一些无用的cuda相关包，而且体积超大；
# 可以直接下cpu环境的torch包安装
如：https://download.pytorch.org/whl/cpu/torch-2.7.0%2Bcpu-cp311-cp311-manylinux_2_28_x86_64.whl
若是dockerfile中，则不同包不要分层安装，应该一起安装，包含CPU环境的torch;
如：
RUN cd /root/whl && pip install --no-cache-dir \
    torch-2.7.0+cpu-cp311-cp311-manylinux_2_28_x86_64.whl certifi-2024.8.30-py3-none-any.whl click-8.1.8-py3-none-any.whl langgraph-0.3.27-py3-none-any.whl presto_python_client-0.8.4-py3-none-any.whl six-1.17.0-py2.py3-none-any.whl charset_normalizer-2.1.1-py3-none-any.whl idna-3.9-py3-none-any.whl polib-1.2.0-py2.py3-none-any.whl requests-2.32.3-py3-none-any.whl urllib3-2.3.0-py3-none-any.whl langchain_core-0.3.59-py3-none-any.whl langgraph_checkpoint-2.0.25-py3-none-any.whl langgraph_prebuilt-0.1.8-py3-none-any.whl langchain==0.3.23 langchain-openai==0.3.12 langchain-anthropic==0.3.12 langchain-community==0.3.21 langchain-deepseek==0.1.3 langchain-google-genai==2.1.2 langchain-mcp-adapters==0.0.5 langchain-huggingface==0.1.2 langchain-ollama==0.3.2 django==4.2.20 djangorestframework==3.16.0 mcp==1.4.1 python-docx==1.1.2 pymysql==1.1.1 accelerate==1.6.0  && cd .. && rm -rf whl && rm -rf /root/.cache

