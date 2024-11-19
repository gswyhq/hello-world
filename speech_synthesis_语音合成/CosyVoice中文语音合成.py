#!/usr/lib/python3
# -*- coding: utf-8 -*-

# 环境要求python3.8

# 第一步，下载代码库：
root@dsw-593648-559cb4b449-6xx55:/mnt/workspace/speech_tts# git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git
# If you failed to clone submodule due to network failures, please run following command until success
cd CosyVoice
git submodule update --init --recursive
pip3 install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com

# 注意，若是CPU环境，需要修改requirements.txt中的GPU库

# 第二步，下载预训练的模型：
注意代码执行目录为CosyVoice目录，即：root@dsw-593648-559cb4b449-6xx55:/mnt/workspace/speech_tts/CosyVoice

from modelscope import snapshot_download
snapshot_download('iic/CosyVoice-300M', cache_dir='pretrained_models/CosyVoice-300M')
snapshot_download('iic/CosyVoice-300M-SFT', cache_dir='pretrained_models/CosyVoice-300M-SFT')
snapshot_download('iic/CosyVoice-300M-Instruct', cache_dir='pretrained_models/CosyVoice-300M-Instruct')
snapshot_download('speech_tts/speech_kantts_ttsfrd', cache_dir='pretrained_models/speech_kantts_ttsfrd')
snapshot_download('iic/CosyVoice-ttsfrd', cache_dir='pretrained_models/CosyVoice-ttsfrd')

# 下载完后目录结构是下面这样：
root@dsw-593648-559cb4b449-6xx55:/mnt/workspace/speech_tts/CosyVoice/pretrained_models# tree
.
├── CosyVoice-300M
│   ├── campplus.onnx
│   ├── configuration.json
│   ├── cosyvoice.yaml
│   ├── flow.pt
│   ├── hift.pt
│   ├── llm.pt
│   ├── README.md
│   ├── speech_tokenizer_v1.onnx
│   └── spk2info.pt
├── CosyVoice-300M-Instruct
│   ├── campplus.onnx
│   ├── configuration.json
│   ├── cosyvoice.yaml
│   ├── flow.pt
│   ├── hift.pt
│   ├── llm.pt
│   ├── README.md
│   ├── speech_tokenizer_v1.onnx
│   └── spk2info.pt
├── CosyVoice-300M-SFT
│   ├── campplus.onnx
│   ├── configuration.json
│   ├── cosyvoice.yaml
│   ├── flow.pt
│   ├── hift.pt
│   ├── llm.pt
│   ├── README.md
│   ├── speech_tokenizer_v1.onnx
│   └── spk2info.pt
├── CosyVoice-ttsfrd
│   ├── configuration.json
│   ├── README.md
│   ├── resource
│   │   ├── festival
│   │   │   ├── apml_f2bf0lr.scm
│   │   │   ├── apml_kaldurtreeZ.scm
│   │   │   ├── apml.scm
│   │   │   ├── cart_aux.scm
│   │   │   ├── clunits_build.scm
...


# 设置依赖包路径：
!export PYTHONPATH=/mnt/workspace/speech_tts/CosyVoice/third_party/Matcha-TTS
或者：
import sys
sys.path.append('/mnt/workspace/speech_tts/CosyVoice')
sys.path.append('/mnt/workspace/speech_tts/CosyVoice/third_party/Matcha-TTS')

# 加载模型进行中文语音生成：
import time
from cosyvoice.cli.cosyvoice import CosyVoice
from cosyvoice.utils.file_utils import load_wav
import torchaudio
start_time = time.time()
cosyvoice = CosyVoice('pretrained_models/CosyVoice-300M-SFT')
# sft usage
print(cosyvoice.list_avaliable_spks())  # 支持的语音人列表：['中文女', '中文男', '日语男', '粤语女', '英文女', '英文男', '韩语女']
output = cosyvoice.inference_sft('你好，我是通义生成式语音大模型，请问有什么可以帮您的吗？', '中文女')
torchaudio.save('sft.wav', output['tts_speech'], 22050)

cosyvoice = CosyVoice('pretrained_models/CosyVoice-300M')
# zero_shot usage
prompt_speech_16k = load_wav('/mnt/workspace/speech_tts/CosyVoice/zero_shot_prompt.wav', 16000)
output = cosyvoice.inference_zero_shot('收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。', '希望你以后能够做的比我还好呦。', prompt_speech_16k)
torchaudio.save('zero_shot.wav', output['tts_speech'], 22050)
# cross_lingual usage
prompt_speech_16k = load_wav('/mnt/workspace/speech_tts/CosyVoice/cross_lingual_prompt.wav', 16000)
output = cosyvoice.inference_cross_lingual('<|en|>And then later on, fully acquiring that company. So keeping management in line, interest in line with the asset that\'s coming into the family is a reason why sometimes we don\'t buy the whole thing.', prompt_speech_16k)
torchaudio.save('cross_lingual.wav', output['tts_speech'], 22050)

cosyvoice = CosyVoice('pretrained_models/CosyVoice-300M-Instruct')
# instruct usage
output = cosyvoice.inference_instruct('在面对挑战时，他展现了非凡的<strong>勇气</strong>与<strong>智慧</strong>。', '中文男', 'Theo \'Crimson\', is a fiery, passionate rebel leader. Fights with fervor for justice, but struggles with impulsiveness.')
torchaudio.save('instruct.wav', output['tts_speech'], 22050)

end_time = time.time()

print("总共耗时：（秒）", end_time - start_time)
# 总共耗时：（秒） 1507.0422947406769

# 总结：效果孩可以，但太慢了，在8核32GB内存的CPU机器上，上面程序运行了25分钟。

def main():
    pass


if __name__ == '__main__':
    main()
