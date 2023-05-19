#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
USERNAME = os.getenv("USERNAME")
# 模型来源：https://huggingface.co/facebook/tts_transformer-zh-cv7_css10
model_dir = rf"D:\Users\{USERNAME}\data\tts-transformer-zh-cv7-css10"

from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub, load_model_ensemble_and_task, Path
from fairseq.models.text_to_speech.hub_interface import TTSHubInterface
import soundfile
import IPython.display as ipd


# models, cfg, task = load_model_ensemble_and_task_from_hf_hub(
#     model_dir,
#     arg_overrides={"vocoder": "hifigan", "fp16": False}
# )
arg_overrides={"vocoder": "hifigan", "fp16": False,
               "data": model_dir}
models, cfg, task = load_model_ensemble_and_task(
        [p.as_posix() for p in Path(model_dir).glob("*.pt")],
        arg_overrides=arg_overrides,
    )
model = models
TTSHubInterface.update_cfg_with_data_cfg(cfg, task.data_cfg)
generator = task.build_generator(model, cfg)

text = "好好学习，天天向上。"

sample = TTSHubInterface.get_model_input(task, text)
wav, sample_rate = TTSHubInterface.get_prediction(task, models[0], generator, sample)

save_wav_file = rf"D:/Users/{USERNAME}/data/tts-transformer-zh-cv7-css10/test_output.wav"
soundfile.write(save_wav_file, wav, sample_rate, 'PCM_16') # 保存wav到文件

ipd.Audio(wav, rate=rate) # 播放wav

def main():
    pass


if __name__ == '__main__':
    main()
