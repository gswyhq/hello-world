#!/usr/bin/env python
# coding=utf-8

import os
import sys
import torch
import soundfile
import numpy as np
import pandas as pd
import io
import base64

sys.path.append("/github_project/ChatTTS") # https://github.com/2noise/ChatTTS
model_dir = "/modelscope/chatTTS" #  https://www.modelscope.cn/pzc163/chatTTS.git
spk_emb_file = rf"/github_project/ChatTTS_Speaker/evaluation_results.csv" # 发音人文件；https://github.com/6drf21e/ChatTTS_Speaker.git

import ChatTTS


seed = 1579  # 根据csv内容，选择一个音色ID

spk_emb_df = pd.read_csv(spk_emb_file, encoding="utf-8")

chat = ChatTTS.Chat()
chat.load_models(compile=False, source='local', local_path=model_dir) # Set to True for better performance

def deterministic(seed=0):
    """
    Set random seed for reproducibility
    :param seed:
    :return: 固定音色向量
    """
    # ref: https://github.com/Jackiexiao/ChatTTS-api-ui-docker/blob/main/api.py#L27
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    seed_id = f"seed_{seed}"
    row = spk_emb_df[spk_emb_df["seed_id"] == seed_id]
    if row.empty:
        print(f"选择的seed无效：{seed}")
    row = row.iloc[0]
    base64_str = row["emb_data"]
    emb_data = base64.b64decode(base64_str)
    spk_emb = torch.load(io.BytesIO(emb_data), map_location=torch.device('cpu'))
    return spk_emb

texts = ['大悲大喜之间，秦尘浑身飘飘欲仙，就想在这里昏睡过去。',
    '但他知道，自己决不能睡。',
    '必须趁热打铁，凝练真气。',
    '甚至于重伤的身体，也只能以后再想办法恢复。',
    '“九星神帝诀，是我从神禁之地得到的最神秘功法，让我看看，此功法究竟有何玄妙。”',
    '秦尘强忍着昏沉之意，运转九星神帝诀中的心法口诀。',
    '这一运转，一股清凉的感觉瞬间传遍他的全身，整个人猛地一个激灵。',
    '原本的昏沉之意，神奇般的瞬间消失。',
    '整个修炼室中，大量的天地元气仿佛被无形的力量牵引，渗入他身体的每一个毛孔之中。',
    '本来真气已经干涸的经脉毛孔，便如久旱逢甘雨一样，贪婪吸收着周围的天地元气。',
    '仅仅片刻的功夫，秦尘的经脉中，已经出现了一丝丝的真气流转。',
    '“好快的真气凝聚速度，比我前世修炼的天级功法，都要快上一倍不止，而且十二条经脉同时吸收，这速度，简直逆天。”',
]


rand_spk = deterministic(seed)
params_infer_code = {
    'prompt': '[speed_5]',
    'spk_emb': rand_spk, # add sampled speaker
    'temperature': .01, # 默认是.3 using custom temperature, 较低的温度使分布更为尖锐，减少随机性，使高概率词更可能被选中
    'top_P': 0.7, # 默认0.7 ；top P decode，控制候选词集合大小，累积概率阈值
    'top_K': 20, # 默认20， top K decode，从概率最高的k个词中选中下一个；
}

params_refine_text = {
  'prompt': '[oral_0][laugh_0][break_0]'
}


wavs = chat.infer(texts, params_refine_text=params_refine_text, params_infer_code=params_infer_code)

save_wav_file = f"output_seed_{seed}.wav"

soundfile.write(save_wav_file, np.hstack(tuple(t[0] for t in wavs)), 24000, 'PCM_16')

def main():
    pass


if __name__ == "__main__":
    main()
