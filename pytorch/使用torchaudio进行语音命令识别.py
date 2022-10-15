#!/usr/bin/env python
# coding: utf-8

# In[ ]:

# https://pytorch.org/tutorials/intermediate/speech_command_classification_with_torchaudio_tutorial.html

# Uncomment the line corresponding to your "runtime type" to run in Google Colab

# CPU:
# !pip install pydub torch==1.7.0+cpu torchvision==0.8.1+cpu torchaudio==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html

# GPU:
# !pip install pydub torch==1.7.0+cu101 torchvision==0.8.1+cu101 torchaudio==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
import sys

import matplotlib.pyplot as plt
import IPython.display as ipd

from tqdm import tqdm


# Let’s check if a CUDA GPU is available and select our device. Running
# the network on a GPU will greatly decrease the training/testing runtime.
# 
# 
# 

# In[ ]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# 导入数据集
# 我们使用 torchaudio 来下载和表示数据集。在这里，我们使用 SpeechCommands，它是由不同人说出的 35 个命令的数据集。数据集 SPEECHCOMMANDS是数据集的一个torch.utils.data.Dataset版本。在这个数据集中，所有的音频文件都大约 1 秒长（因此大约 16000 个时间帧长）。
#
# 实际的加载和格式化步骤发生在访问数据点时，torchaudio 负责将音频文件转换为张量。如果想直接加载音频文件， torchaudio.load()可以使用。它返回一个元组，其中包含新创建的张量以及音频文件的采样频率（SpeechCommands 为 16kHz）。
#
# 回到数据集，在这里我们创建一个子类，将其拆分为标准训练、验证、测试子集。


from torchaudio.datasets import SPEECHCOMMANDS
import os


class SubsetSC(SPEECHCOMMANDS):
    def __init__(self, subset: str = None):
        super().__init__("./", download=True)

        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath) as fileobj:
                return [os.path.normpath(os.path.join(self._path, line.strip())) for line in fileobj]

        if subset == "validation":
            self._walker = load_list("validation_list.txt")
        elif subset == "testing":
            self._walker = load_list("testing_list.txt")
        elif subset == "training":
            excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
            excludes = set(excludes)
            self._walker = [w for w in self._walker if w not in excludes]


# Create training and testing split of the data. We do not use validation in this tutorial.
# 下载英文指令音频文件（http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz），文件大小共2.3GB
train_set = SubsetSC("training")
test_set = SubsetSC("testing")

waveform, sample_rate, label, speaker_id, utterance_number = train_set[0]


# SPEECHCOMMANDS 数据集中的数据点是由波形（音频信号）、采样率、话语（标签）、说话者 ID、话语数组成的元组。

# In[ ]:


print("Shape of waveform: {}".format(waveform.size()))
print("Sample rate of waveform: {}".format(sample_rate))

plt.plot(waveform.t().numpy());


# 让我们找到数据集中可用的标签列表。
labels = sorted(list(set(datapoint[2] for datapoint in train_set)))
labels


# 35 个音频标签是用户说出的命令。前几个文件是人们说“marvin”.
waveform_first, *_ = train_set[0]
ipd.Audio(waveform_first.numpy(), rate=sample_rate)

waveform_second, *_ = train_set[1]
ipd.Audio(waveform_second.numpy(), rate=sample_rate)


# 最后一个文件是说 “visual”.

# In[ ]:


waveform_last, *_ = train_set[-1]
ipd.Audio(waveform_last.numpy(), rate=sample_rate)


# 格式化数据
# 这是对数据应用转换的好地方。对于波形，我们对音频进行下采样以加快处理速度，而不会损失太多的分类能力。
#
# 我们不需要在这里应用其他转换。某些数据集通常必须通过沿通道维度取平均值或仅保留一个通道来减少通道数量（例如从立体声到单声道）。由于 SpeechCommands 使用单个音频通道，因此这里不需要。


new_sample_rate = 8000
transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=new_sample_rate)
transformed = transform(waveform)

ipd.Audio(transformed.numpy(), rate=new_sample_rate)


# 我们使用标签列表中的索引对每个单词进行编码。

def label_to_index(word):
    # Return the position of the word in labels
    return torch.tensor(labels.index(word))


def index_to_label(index):
    # Return the word corresponding to the index in labels
    # This is the inverse of label_to_index
    return labels[index]


word_start = "yes"
index = label_to_index(word_start)
word_recovered = index_to_label(index)

print(word_start, "-->", index, "-->", word_recovered)


# 为了将由音频记录和话语组成的数据点列表转换为模型的两个批量张量，我们实现了一个由 PyTorch DataLoader 使用的 collat​​e 函数，该函数允许我们批量迭代数据集。
#
# 在 collat​​e 函数中，我们还应用了重采样和文本编码。

def pad_sequence(batch):
    # Make all tensor in a batch the same length by padding with zeros
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    return batch.permute(0, 2, 1)


def collate_fn(batch):

    # A data tuple has the form:
    # waveform, sample_rate, label, speaker_id, utterance_number

    tensors, targets = [], []

    # Gather in lists, and encode labels as indices
    for waveform, _, label, *_ in batch:
        tensors += [waveform]
        targets += [label_to_index(label)]

    # Group the list of tensors into a batched tensor
    tensors = pad_sequence(tensors)
    targets = torch.stack(targets)

    return tensors, targets


batch_size = 256

if device == "cuda":
    num_workers = 1
    pin_memory = True
else:
    num_workers = 0
    pin_memory = False

train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=num_workers,
    pin_memory=pin_memory,
)
test_loader = torch.utils.data.DataLoader(
    test_set,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    collate_fn=collate_fn,
    num_workers=num_workers,
    pin_memory=pin_memory,
)


# 定义网络
# 在本教程中，我们将使用卷积神经网络来处理原始音频数据。通常对音频数据应用更高级的转换，但 CNN 可用于准确处理原始数据。
# 具体架构仿照本文描述的M5网络架构。模型处理原始音频数据的一个重要方面是其第一层过滤器的感受野。
# 我们模型的第一个滤波器长度为 80，因此在处理以 8kHz 采样的音频时，感受野约为 10ms（在 4kHz 时约为 20ms）。
# 这个大小类似于经常使用从 20 毫秒到 40 毫秒的感受野的语音处理应用程序。

class M5(nn.Module):
    def __init__(self, n_input=1, n_output=35, stride=16, n_channel=32):
        super().__init__()
        self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=80, stride=stride)
        self.bn1 = nn.BatchNorm1d(n_channel)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(n_channel, n_channel, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(n_channel)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(n_channel, 2 * n_channel, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(2 * n_channel)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(2 * n_channel)
        self.pool4 = nn.MaxPool1d(4)
        self.fc1 = nn.Linear(2 * n_channel, n_output)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool4(x)
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        return F.log_softmax(x, dim=2)


model = M5(n_input=transformed.shape[0], n_output=len(labels))
model.to(device)
print(model)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


n = count_parameters(model)
print("Number of parameters: %s" % n)


# 我们将使用本文中使用的相同优化技术，即权重衰减设置为 0.0001 的 Adam 优化器。
# 起初，我们将以 0.01 的学习率进行训练，但scheduler在 20 个 epoch 之后的训练期间，我们将使用 a 将其降低到 0.001。

optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)  # reduce the learning after 20 epochs by a factor of 10


# 训练和测试网络
# 现在让我们定义一个训练函数，它将我们的训练数据输入模型并执行反向传递和优化步骤。对于训练，我们将使用的损失是负对数似然。
# 然后将在每个 epoch 之后对网络进行测试，以查看在训练期间准确性如何变化。

def train(model, epoch, log_interval):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):

        data = data.to(device)
        target = target.to(device)

        # apply transform and model on whole batch directly on device
        data = transform(data)
        output = model(data)

        # negative log-likelihood for a tensor of size (batch x 1 x n_output)
        loss = F.nll_loss(output.squeeze(), target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print training stats
        if batch_idx % log_interval == 0:
            print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")

        # update progress bar
        pbar.update(pbar_update)
        # record loss
        losses.append(loss.item())


# 现在我们有了一个训练函数，我们需要制作一个来测试网络的准确性。
# 我们将模型设置为eval()模式，然后在测试数据集上运行推理。
# 调用eval()将网络中所有模块中的训练变量设置为 false。
# 某些层（如批量标准化和 dropout 层）在训练期间表现不同，因此这一步对于获得正确结果至关重要。

def number_of_correct(pred, target):
    # count number of correct predictions
    return pred.squeeze().eq(target).sum().item()


def get_likely_index(tensor):
    # find most likely label index for each element in the batch
    return tensor.argmax(dim=-1)


def test(model, epoch):
    model.eval()
    correct = 0
    for data, target in test_loader:

        data = data.to(device)
        target = target.to(device)

        # apply transform and model on whole batch directly on device
        data = transform(data)
        output = model(data)

        pred = get_likely_index(output)
        correct += number_of_correct(pred, target)

        # update progress bar
        pbar.update(pbar_update)

    print(f"\nTest Epoch: {epoch}\tAccuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n")


# 最后，我们可以训练和测试网络。我们将网络训练 10 个 epoch，然后降低学习率并再训练 10 个 epoch。
# 网络将在每个 epoch 之后进行测试，以查看在训练期间准确度如何变化。

log_interval = 20
n_epoch = 2

pbar_update = 1 / (len(train_loader) + len(test_loader))
losses = []

# The transform needs to live on the same device as the model and the data.
transform = transform.to(device)
with tqdm(total=n_epoch) as pbar:
    for epoch in range(1, n_epoch + 1):
        train(model, epoch, log_interval)
        test(model, epoch)
        scheduler.step()

# Let's plot the training loss versus the number of iteration.
# plt.plot(losses);
# plt.title("training loss");


# 在 2 个 epoch 之后，网络在测试集上的准确率应该超过 65%，在 21 个 epoch 之后应该达到 85%。
# 让我们看看训练集中的最后一句话，看看模型是如何处理它的。


def predict(tensor):
    # Use the model to predict the label of the waveform
    tensor = tensor.to(device)
    tensor = transform(tensor)
    tensor = model(tensor.unsqueeze(0))
    tensor = get_likely_index(tensor)
    tensor = index_to_label(tensor.squeeze())
    return tensor


waveform, sample_rate, utterance, *_ = train_set[-1]
ipd.Audio(waveform.numpy(), rate=sample_rate)

print(f"Expected: {utterance}. Predicted: {predict(waveform)}.")


# 让我们找一个分类不正确的例子，如果有的话。


for i, (waveform, sample_rate, utterance, *_) in enumerate(test_set):
    output = predict(waveform)
    if output != utterance:
        ipd.Audio(waveform.numpy(), rate=sample_rate)
        print(f"Data point #{i}. Expected: {utterance}. Predicted: {output}.")
        break
    else:
        print("All examples in this dataset were correctly classified!")
        print("In this case, let's just look at the last data point")
        ipd.Audio(waveform.numpy(), rate=sample_rate)
        print(f"Data point #{i}. Expected: {utterance}. Predicted: {output}.")


# 随意尝试使用您自己的其中一个标签的录音！
# 例如，使用 Colab，在执行下面的单元格时说“Go”。这将记录一秒钟的音频并尝试对其进行分类。

def record(seconds=1):

    from google.colab import output as colab_output
    from base64 import b64decode
    from io import BytesIO
    from pydub import AudioSegment

    RECORD = (
        b"const sleep  = time => new Promise(resolve => setTimeout(resolve, time))\n"
        b"const b2text = blob => new Promise(resolve => {\n"
        b"  const reader = new FileReader()\n"
        b"  reader.onloadend = e => resolve(e.srcElement.result)\n"
        b"  reader.readAsDataURL(blob)\n"
        b"})\n"
        b"var record = time => new Promise(async resolve => {\n"
        b"  stream = await navigator.mediaDevices.getUserMedia({ audio: true })\n"
        b"  recorder = new MediaRecorder(stream)\n"
        b"  chunks = []\n"
        b"  recorder.ondataavailable = e => chunks.push(e.data)\n"
        b"  recorder.start()\n"
        b"  await sleep(time)\n"
        b"  recorder.onstop = async ()=>{\n"
        b"    blob = new Blob(chunks)\n"
        b"    text = await b2text(blob)\n"
        b"    resolve(text)\n"
        b"  }\n"
        b"  recorder.stop()\n"
        b"})"
    )
    RECORD = RECORD.decode("ascii")

    print(f"Recording started for {seconds} seconds.")
    display(ipd.Javascript(RECORD))
    s = colab_output.eval_js("record(%d)" % (seconds * 1000))
    print("Recording ended.")
    b = b64decode(s.split(",")[1])

    fileformat = "wav"
    filename = f"_audio.{fileformat}"
    AudioSegment.from_file(BytesIO(b)).export(filename, format=fileformat)
    return torchaudio.load(filename)


# Detect whether notebook runs in google colab
if "google.colab" in sys.modules:
    waveform, sample_rate = record()
    print(f"Predicted: {predict(waveform)}.")
    ipd.Audio(waveform.numpy(), rate=sample_rate)


# 结论
# 在本教程中，我们使用 torchaudio 加载数据集并重新采样信号。
# 然后，我们定义了一个神经网络，我们训练它来识别给定的命令。
# 还有其他数据预处理方法，例如找到梅尔频率倒谱系数（MFCC），可以减少数据集的大小。
# 这种变换也可以在 torchaudio 中作为torchaudio.transforms.MFCC.


