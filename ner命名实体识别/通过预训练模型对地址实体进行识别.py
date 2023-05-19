#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, time
import json
USERNAME = os.getenv('USERNAME')
import pandas as pd
import torch
from transformers import pipeline
from transformers import AutoTokenizer, BertForTokenClassification
from transformers.pipelines.token_classification import TokenClassificationPipeline
from transformers import AutoConfig, AutoTokenizer, BertForTokenClassification

# # https://huggingface.co/jiaqianjing/chinese-address-ner
model_dir = f'D:\\Users\\{USERNAME}\\data\\chinese-address-ner'

tokenizer = AutoTokenizer.from_pretrained(model_dir)
ner = pipeline('ner', model=model_dir)
text = "北京市海淀区西北旺东路10号院(马连洼街道西北旺社区东北方向)"
output_list = ner(text)

label_dict = {'LABEL_0': ('LABEL_0'),
 'LABEL_1': ('LABEL_1', 'LABEL_2'),
 'LABEL_2': ('LABEL_1', 'LABEL_2'),
 'LABEL_3': ('LABEL_3', 'LABEL_4'),
 'LABEL_4': ('LABEL_3', 'LABEL_4'),
 'LABEL_5': ('LABEL_5', 'LABEL_6'),
 'LABEL_6': ('LABEL_5', 'LABEL_6'),
 'LABEL_7': ('LABEL_7', 'LABEL_8'),
 'LABEL_8': ('LABEL_7', 'LABEL_8'),
 'LABEL_9': ('LABEL_9', 'LABEL_10'),
 'LABEL_10': ('LABEL_9', 'LABEL_10'),
 'LABEL_11': ('LABEL_11', 'LABEL_12'),
 'LABEL_12': ('LABEL_11', 'LABEL_12'),
 'LABEL_13': ('LABEL_13', 'LABEL_14'),
 'LABEL_14': ('LABEL_13', 'LABEL_14')}

def output2dict(output_list):
    result_list = []
    last_entity = None
    for ret in output_list:
        entity = ret['entity']
        word = ret['word']
        if label_dict.get(last_entity) and entity in label_dict[last_entity]:
            result_list[-1] = (result_list[-1][0]+word, entity)
        else:
            last_entity = entity
            result_list.append((word, entity))
    return result_list

# [('北京市', 'LABEL_2'),
#  ('海淀区', 'LABEL_6'),
#  ('西北旺东路10号院', 'LABEL_14'),
#  ('(', 'LABEL_0'),
#  ('马连洼街道', 'LABEL_8'),
#  ('西北旺社区', 'LABEL_10'),
#  ('东北方向)', 'LABEL_0')]

# 按照行政级别（总有 7 级）抽取地址信息
# 1：省 2：市 3：区 4：乡镇街道 5：乡村社区 6：楼宇大厦学校公园 7：道路+道路号
# 返回类别	解释
# LABEL_0	忽略信息
# LABEL_1	第1级地址（头）
# LABEL_2	第1级地址（其余部分）
# LABEL_3	第2级地址（头）
# LABEL_4	第2级地址（其余部分）
# LABEL_5	第3级地址（头）
# LABEL_6	第3级地址（其余部分）
# LABEL_7	第4级地址（头）
# LABEL_8	第4级地址（其余部分）
# LABEL_9	第5级地址（头）
# LABEL_10	第5级地址（其余部分）
# LABEL_11	第6级地址（头）
# LABEL_12	第6级地址（其余部分）
# LABEL_13	第7级地址（头）
# LABEL_14	第7级地址（其余部分）

#########################################################################################################################
text = '沙井镇安郎路F9号(马安山地铁站C口步行340米)'
tokenizer = AutoTokenizer.from_pretrained(model_dir)
config = AutoConfig.from_pretrained(model_dir)
model = BertForTokenClassification(config=config).from_pretrained(model_dir)
token_classifier = TokenClassificationPipeline(model=model, tokenizer=tokenizer, task='ner', )
output2dict(token_classifier(text))

# [('沙井镇', 'LABEL_8'),
#  ('安郎路f##9号', 'LABEL_14'),
#  ('(', 'LABEL_0'),
#  ('马安山地铁站', 'LABEL_14'),
#  ('c口步行340米)', 'LABEL_0')]

######################################### 识别的实体带有#号 还原为原有实体 #################################################################################
# 识别的实体带有#号 还原为原有实体
model_inputs = tokenizer(text, return_offsets_mapping=True, return_tensors='pt' )
offset_mapping = model_inputs.pop('offset_mapping')
output = model(**model_inputs)
label_list = [model.config.id2label[idx.tolist()] for idx in output.logits[0].argmax(dim=-1)]
def label_offest_to_dict(text, label_list, offset_mapping):
    '''通过offset_mapping定位，还原原有字符串内容，避免出现##号'''
    result_list = []
    last_entity = None
    for (start, end), entity in zip(offset_mapping, label_list):
        word = text[start: end]
        if label_dict.get(last_entity) and entity in label_dict[last_entity]:
            result_list[-1] = (result_list[-1][0] + word, entity)
        else:
            last_entity = entity
            result_list.append((word, entity))
    return result_list

label_offest_to_dict(text, label_list[1:-1], offset_mapping[0][1:-1])

# [('沙井镇', 'LABEL_8'),
#  ('安郎路F9号', 'LABEL_14'),
#  ('(', 'LABEL_0'),
#  ('马安山地铁站', 'LABEL_14'),
#  ('C口步行340米)', 'LABEL_0')]

############################################### 量化为float16 #########################################################################
# 对bert模型进行量化，这里仅仅量化 Embedding、Linear层：
# 但经过测试，量化为int8，对结果影响较大；

qconfig_dict = {
    torch.nn.Embedding: torch.quantization.float_qparams_weight_only_qconfig,
    torch.nn.Linear: torch.quantization.float16_dynamic_qconfig,
}
quantized_model = torch.quantization.quantize_dynamic(model, qconfig_dict, dtype=torch.float16)

# 使用量化后的模型：
token_classifier = TokenClassificationPipeline(model=quantized_model, tokenizer=tokenizer, task='ner',)
output2dict(token_classifier(text))

# 保存量化的模型：
torch.save(quantized_model, f'D:\\Users\\{USERNAME}\\data\\chinese-address-ner\\model_quant')

#########################################################################################################################

df = pd.read_excel(rf"D:\Users\{USERNAME}\data\数据梳理\经纬度地理位置\20230315\2000条地址_20230317.xlsx", header=None, names=['原始测试地址'])

addr_ret_list1 = []
token_classifier = TokenClassificationPipeline(model=model, tokenizer=tokenizer, task='ner', )
start_time = time.time()
for text in df['原始测试地址'].values:
    if text and text.strip():
        ret = output2dict(token_classifier(text.strip()))
    else:
        ret = []
    addr_ret_list1.append(json.dumps(ret, ensure_ascii=False))
    print(text)
    print(ret)
    print('-'*20)

use_time = time.time() - start_time
print(f'原始模型总耗时：{round(use_time, 6)}秒, 平均耗时：{round(use_time/df.shape[0], 6)}秒', )
df['原始模型结果'] = addr_ret_list1
df.to_excel(rf"D:\Users\{USERNAME}\data\数据梳理\经纬度地理位置\20230315\2000条地址_预训练模型识别结果.xlsx", index=False)
# 原始模型总耗时：232.477324秒, 平均耗时：0.104909秒


addr_ret_list1 = []
model = torch.load(f'D:\\Users\\{USERNAME}\\data\\chinese-address-ner\\model_quant')
token_classifier = TokenClassificationPipeline(model=model, tokenizer=tokenizer, task='ner', )
start_time = time.time()
for text in df['原始测试地址'].values:
    if text and text.strip():
        ret = output2dict(token_classifier(text.strip()))
    else:
        ret = []
    addr_ret_list1.append(json.dumps(ret, ensure_ascii=False))
    print(text)
    print(ret)
    print('-'*20)

use_time = time.time() - start_time
print(f'量化模型总耗时：{round(use_time, 6)}秒, 平均耗时：{round(use_time/df.shape[0], 6)}秒', )
df['量化模型结果'] = addr_ret_list1
df.to_excel(rf"D:\Users\{USERNAME}\data\数据梳理\经纬度地理位置\20230315\2000条地址_预训练模型识别结果.xlsx", index=False)

# 原始模型总耗时：232.477324秒, 平均耗时：0.104909秒
# 量化模型总耗时：373.579705秒, 平均耗时：0.168583秒
# 量化模型耗时竟然比未量化耗时更长；

# 还存在的问题列表：
# 缺失省份，对城市识别存在问题；如：佛山市南海区虹岭路一号 -> [["佛", "LABEL_1"], ["山市", "LABEL_4"], ["南海区", "LABEL_6"], ["虹岭路一号", "LABEL_14"]]
# 缺失街道，对社区名称识别存在问题；如：海乐社区富怡花园1栋105-13 -> [('海', 'LABEL_7'), ('乐社区', 'LABEL_10'), ('富怡花园1栋', 'LABEL_14'), ('105-13', 'LABEL_0')]
# 多个区时候，识别存在问题，如：新安街道海乐社区43区安乐三街152号 -> [('新安街道', 'LABEL_8'), ('海乐社区', 'LABEL_10'), ('43', 'LABEL_0'), ('区安乐三街152号', 'LABEL_14')]
# 无省市区，以街名称开头，存在问题，如：怡康街鸿都商务大厦(顺和路)西侧 -> [('怡', 'LABEL_7'), ('康', 'LABEL_12'), ('街', 'LABEL_8'), ('鸿都商务大厦', 'LABEL_14'), ('(', 'LABEL_0'), ('顺和路', 'LABEL_14'), (')西侧', 'LABEL_0')]
# 街道名称、社区名称无法区分开来，如： 新安街道海乐社区兴华一路北14巷9号102铺-生煎包 -> [('新安街道', 'LABEL_8'), ('海', 'LABEL_9'), ('乐社区', 'LABEL_8'), ('兴华一路北14巷9号', 'LABEL_14'), ('102铺-生', 'LABEL_0'), ('煎包', 'LABEL_14')]
# 新安街道海乐社区兴华一路北十三巷9号9-1 -> [('新安街道', 'LABEL_8'), ('海乐社区兴华一路', 'LABEL_10'), ('北十三巷9号', 'LABEL_14'), ('9-1', 'LABEL_0')]

# 可针对错误数据，利用 全国省会城市POI数据（https://github.com/Pyjacc/data-poi） 进一步微调模型；

def main():
    pass


if __name__ == '__main__':
    main()
