fish-speech

git clone https://github.com/fishaudio/fish-speech

root@dsw-605246-5c744bbc5d-twn88:/mnt/workspace# git clone https://github.com/fishaudio/fish-speech
root@dsw-605246-5c744bbc5d-twn88:/mnt/workspace/fish-speech# cd fish-speech
root@dsw-605246-5c744bbc5d-twn88:/mnt/workspace/fish-speech# pip3 install torch torchvision torchaudio
root@dsw-605246-5c744bbc5d-twn88:/mnt/workspace/fish-speech# pip3 install -e .
root@dsw-605246-5c744bbc5d-twn88:/mnt/workspace/fish-speech# apt install libsox-dev
root@dsw-605246-5c744bbc5d-twn88:/mnt/workspace/fish-speech# git log|head
commit 024667c4a6f3ec732c43dd8d8a45db841ffa8e71


!set HF_ENDPOINT=https://hf-mirror.com
!export HF_ENDPOINT=https://hf-mirror.com

!huggingface-cli download fishaudio/fish-speech-1.2-sft --local-dir checkpoints/fish-speech-1.2-sft/

!python tools/webui.py \
    --llama-checkpoint-path checkpoints/fish-speech-1.2-sft \
    --decoder-checkpoint-path checkpoints/fish-speech-1.2-sft/firefly-gan-vq-fsq-4x1024-42hz-generator.pth
    --device cpu

# 若允许其他机器访问服务，则需要修改webui.py文件，如下：
# sed -i 's/app.launch(/app.launch(server_name="0.0.0.0",/g' tools/webui.py 

# 浏览器打开：http://localhost:7860 即可进行语音合成
# 合成速度极慢：200个字，用时超30min.

docker pull registry.cn-shenzhen.aliyuncs.com/gswyhq/fish-speech:1.2

