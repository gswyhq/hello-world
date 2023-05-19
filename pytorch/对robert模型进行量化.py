#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 来源： https://huggingface.co/uer/chinese_roberta_L-4_H-128/tree/main

import os
import torch
USERNAME = os.getenv('USERNAME')

model_dir = f'D:\\Users\\{USERNAME}\\data\\chinese_roberta_L-4_H-128'
from transformers import pipeline, BertForMaskedLM, BertTokenizer
unmasker = pipeline('fill-mask', model=model_dir)
unmasker("中国的首都是[MASK]京。")

# --------------------------------------------------------------------------------------------------------------
from transformers.pipelines.fill_mask import FillMaskPipeline
mask_model = BertForMaskedLM.from_pretrained(model_dir)
tokenizer = BertTokenizer.from_pretrained(model_dir)
unmasker = FillMaskPipeline(mask_model, tokenizer)
unmasker("中国的首都是[MASK]京。")

########################################################################################################################
# 对Robert模型进行量化，这里仅仅量化 Embedding、Linear层：
backend = "fbgemm"
qconfig_dict = {
    torch.nn.Embedding: torch.quantization.qconfig.float_qparams_weight_only_qconfig,
    torch.nn.Linear: torch.quantization.get_default_qconfig(backend)
}
qmodel = torch.quantization.quantize_dynamic(mask_model, qconfig_dict, dtype=torch.qint8)

# 使用量化后的模型：
unmasker = FillMaskPipeline(qmodel, tokenizer)
unmasker("中国的首都是[MASK]京。")

# 保存量化的模型：
torch.save(qmodel, f'D:\\Users\\{USERNAME}\\data\\chinese_roberta_L-4_H-128\\model_quant')

############################################### 量化为float16 #########################################################################
# 对bert模型进行量化，这里仅仅量化 Embedding、Linear层：
# 有时候，量化为int8，对结果影响较大；这个时候可以量化为float16
import torch
from transformers.pipelines.token_classification import TokenClassificationPipeline
from transformers import AutoConfig, AutoTokenizer, BertForTokenClassification
model_dir = f'D:\\Users\\{USERNAME}\\data\\chinese-address-ner' # https://huggingface.co/jiaqianjing/chinese-address-ner
tokenizer = AutoTokenizer.from_pretrained(model_dir)
config = AutoConfig.from_pretrained(model_dir)
model = BertForTokenClassification(config=config).from_pretrained(model_dir)

qconfig_dict = {
    torch.nn.Embedding: torch.quantization.float_qparams_weight_only_qconfig,
    torch.nn.Linear: torch.quantization.float16_dynamic_qconfig,
}
qmodel = torch.quantization.quantize_dynamic(model, qconfig_dict, dtype=torch.float16)

# 使用量化后的模型：
token_classifier = TokenClassificationPipeline(model=qmodel, tokenizer=tokenizer, task='ner',)
text = "北京市海淀区西北旺东路10号院(马连洼街道西北旺社区东北方向)"
token_classifier(text)

# 保存量化的模型：
torch.save(qmodel, f'D:\\Users\\{USERNAME}\\data\\chinese-address-ner\\model_quant')

########################################################################################################################
# 当然，若是简单结构的模型，可以通过如下方法量化:
model_fp32.qconfig = torch.quantization.get_default_qconfig('fbgemm')

# 为静态量化准备模型。这会在模型中插入观察器，这些观察器将在校准期间观察激活张量。
model_fp32_prepared = torch.quantization.prepare(model_fp32)

# 校准准备好的模型以确定在真实世界设置中激活的量化参数，校准将使用代表性数据集完成
evaluate(model_fp32_prepared)

# 将观察模型转换为量化模型。
# 量化权重，计算并存储要用于每个激活张量的比例和偏差值，并用量化实现替换关键运算符。
model_int8 = torch.quantization.convert(model_fp32_prepared)
print("model int8", model_int8)
# save model
torch.save(model_int8.state_dict(),"./openpose_vgg_quant.pth")

def main():
    pass


if __name__ == '__main__':
    main()
