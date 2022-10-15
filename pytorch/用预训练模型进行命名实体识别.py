#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# https://huggingface.co/IDEA-CCNL/Erlangshen-Ubert-110M-Chinese/tree/main
# https://github.com/IDEA-CCNL/Fengshenbang-LM/blob/442259fc2519945ba8c71fb4da7137aa8301f68d/fengshen/models/ubert/modeling_ubert.py


import os
from transformers import MegatronBertConfig, MegatronBertModel
from transformers import BertTokenizer
USERNAME = os.getenv("USERNAME")


config = {'architectures': ['BertForMaskedLM'],
 'attention_probs_dropout_prob': 0.1,
 'bos_token_id': 0,
 'directionality': 'bidi',
 'eos_token_id': 2,
 'hidden_act': 'gelu',
 'hidden_dropout_prob': 0.1,
 'hidden_size': 768,
 'biaffine_size': 256,
 'initializer_range': 0.02,
 'intermediate_size': 3072,
 'layer_norm_eps': 1e-12,
 'max_position_embeddings': 512,
 'model_type': 'bert',
 'num_attention_heads': 12,
 'num_hidden_layers': 12,
 'output_past': True,
 'pad_token_id': 1,
 'pooler_fc_size': 768,
 'pooler_num_attention_heads': 12,
 'pooler_num_fc_layers': 3,
 'pooler_size_per_head': 128,
 'pooler_type': 'first_token_transform',
 'type_vocab_size': 2,
 'vocab_size': 21128,
 'pretrained_model_path': rf"D:\Users\{USERNAME}\data\Erlangshen-Ubert-110M-Chinese",
 'load_checkpoints_path': '',
 'monitor': 'train_loss',
 'save_top_k': 3,
 'mode': 'min',
 'every_n_train_steps': 100,
 'save_weights_only': True,
 'checkpoint_path': './checkpoint/',
 'filename': 'model-{epoch:02d}-{train_loss:.4f}',
 'default_root_dir': rf"D:\Users\{USERNAME}\data\Erlangshen-Ubert-110M-Chinese",
 'batchsize': 8,
 'max_length': 128,
 'threshold': 0.5}


class Dict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__

def dict_to_object(dictObj):
    if not isinstance(dictObj, dict):
        return dictObj
    inst = Dict()
    for k, v in dictObj.items():
        inst[k] = dict_to_object(v)
    return inst

from fengshen import UbertPiplines

# 转换字典成为对象，可以用"."方式访问对象属性
args = dict_to_object(config)
test_data = [
    {
        "task_type": "抽取任务",
        "subtask_type": "实体识别",
        "text": "大冲花园房价不低，周边就是高新园地铁站，好多程序员上班。",
        "choices": [
            {"entity_type": "小区名字"},
            {"entity_type": "岗位职责"},
            {'entity_type': '人物姓名'},
             {'entity_type': '岗位职位'},
             {'entity_type': '游戏'},
             {'entity_type': '书名'},
             {'entity_type': '公司'},
             {'entity_type': '组织机构'},
             {'entity_type': '旅游景点'},
             {'entity_type': '地址'},
             {'entity_type': '电影名称'},
             {'entity_type': '政府机构'}
        ],
        "id": 0}
]
model = UbertPiplines(args)
result = model.predict(test_data, cuda=False)
for line in result:
    print(line)

# 抽取结果
# {'task_type': '抽取任务', 'subtask_type': '实体识别', 'text': '大冲花园房价不低，周边就是高新园地铁站，好多程序员上班。', 'choices': [{'entity_type': '小区名字', 'entity_list': [{'entity_name': '大冲花园', 'score': 0.8836232812339245}]}, {'entity_type': '岗位职责', 'entity_list': [{'entity_name': '程序员', 'score': 0.9203148521001129}]}, {'entity_type': '人物姓名', 'entity_list': []}, {'entity_type': '岗位职位', 'entity_list': [{'entity_name': '程序员', 'score': 0.966317785055007}]}, {'entity_type': '游戏', 'entity_list': []}, {'entity_type': '书名', 'entity_list': []}, {'entity_type': '公司', 'entity_list': []}, {'entity_type': '组织机构', 'entity_list': []}, {'entity_type': '旅游景点', 'entity_list': []}, {'entity_type': '地址', 'entity_list': [{'entity_name': '高新园地铁站', 'score': 0.7270529260364157}]}, {'entity_type': '电影名称', 'entity_list': []}, {'entity_type': '政府机构', 'entity_list': []}], 'id': 0}

def main():
    pass


if __name__ == '__main__':
    main()
