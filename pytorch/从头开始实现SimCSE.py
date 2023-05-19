#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from abc import ABC
from tqdm.notebook import tqdm
from dataclasses import dataclass, field
from typing import List, Union, Optional, Dict
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, TrainingArguments, Trainer
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel
from transformers.tokenization_utils_base import PaddingStrategy, PreTrainedTokenizerBase
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithPoolingAndCrossAttentions

# 一、定义参数


@dataclass
class DataArguments:
    train_file: str = field(default="./data/simcse/wiki1m_for_simcse.txt",
                            metadata={"help": "The path of train file"})
    model_name_or_path: str = field(default="E:/pretrained/bert-base-uncased",
                                    metadata={"help": "The name or path of pre-trained language model"})
    max_seq_length: int = field(default=32,
                                metadata={"help": "The maximum total input sequence length after tokenization."})


training_args = TrainingArguments(
    output_dir="./checkpoints",
    num_train_epochs=1,
    per_device_train_batch_size=64,
    learning_rate=3e-5,
    load_best_model_at_end=True,
    overwrite_output_dir=True,
    do_train=True,
    do_eval=False,
    logging_steps=10)

data_args = DataArguments()

# 二、读取数据
# 初始化tokenizer
tokenizer = BertTokenizer.from_pretrained(data_args.model_name_or_path)
# 读取训练数据
with open(data_args.train_file, encoding="utf8") as file:
    texts = [line.strip() for line in tqdm(file.readlines())]
print(type(texts))
print(texts[0])

# 三、构建Dataset和collate_fn
# 3.1 构建Dataset

class PairDataset(Dataset):
    def __init__(self, examples: List[str]):
        total = len(examples)
        # 将所有样本复制一份用于对比学习
        sentences_pair = examples + examples
        sent_features = tokenizer(sentences_pair,
                                  max_length=data_args.max_seq_length,
                                  truncation=True,
                                  padding=False)
        features = {}
        # 将相同的样本放在同一个列表中
        for key in sent_features:
            features[key] = [[sent_features[key][i], sent_features[key][i + total]] for i in tqdm(range(total))]
        self.input_ids = features["input_ids"]
        self.attention_mask = features["attention_mask"]
        self.token_type_ids = features["token_type_ids"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, item):
        return {
            "input_ids": self.input_ids[item],
            "attention_mask": self.attention_mask[item],
            "token_type_ids": self.token_type_ids[item]
        }


train_dataset = PairDataset(texts)
print(train_dataset[0])

{'input_ids': [[101, 26866, 1999, 2148, 2660, 102], [101, 26866, 1999, 2148, 2660, 102]],
 'attention_mask': [[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]], 'token_type_ids': [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]}

# 3.2 构建collate_fn
@dataclass
class DataCollator:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        special_keys = ['input_ids', 'attention_mask', 'token_type_ids']
        batch_size = len(features)
        if batch_size == 0:
            return
        # flat_features: [sen1, sen1, sen2, sen2, ...]
        flat_features = []
        for feature in features:
            for i in range(2):
                flat_features.append({k: feature[k][i] for k in feature.keys() if k in special_keys})
        # padding
        batch = self.tokenizer.pad(
            flat_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        # batch_size, 2, seq_len
        batch = {k: batch[k].view(batch_size, 2, -1) for k in batch if k in special_keys}
        return batch


collate_fn = DataCollator(tokenizer)
dataloader = DataLoader(train_dataset, batch_size=4, collate_fn=collate_fn)
batch = next(iter(dataloader))
print(batch.keys())
print(batch["input_ids"].shape)

dict_keys(['input_ids', 'attention_mask', 'token_type_ids'])
torch.Size([4, 2, 32])

# 四、构建模型


# 全连接层，用于投影CLS的向量表示
class MLPLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.dense = nn.Linear(input_size, output_size)
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)
        return x


# 相似度层，计算向量间相似度
class Similarity(nn.Module):
    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


# SimCSE的完整模型结构
class BertForCL(BertPreTrainedModel, ABC):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.mlp = MLPLayer(config.hidden_size, config.hidden_size)
        self.sim = Similarity(temp=0.05)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                sent_emb=False):
        if sent_emb:
            # 模型推断时使用的forward
            return self.sentemb_forward(input_ids=input_ids,
                                        attention_mask=attention_mask,
                                        token_type_ids=token_type_ids,
                                        position_ids=position_ids,
                                        head_mask=head_mask,
                                        inputs_embeds=inputs_embeds,
                                        labels=labels,
                                        output_attentions=output_attentions,
                                        output_hidden_states=output_hidden_states,
                                        return_dict=return_dict)
        else:
            # 模型训练时使用的forward
            return self.cl_forward(input_ids=input_ids,
                                   attention_mask=attention_mask,
                                   token_type_ids=token_type_ids,
                                   position_ids=position_ids,
                                   head_mask=head_mask,
                                   inputs_embeds=inputs_embeds,
                                   labels=labels,
                                   output_attentions=output_attentions,
                                   output_hidden_states=output_hidden_states,
                                   return_dict=return_dict)

    def sentemb_forward(self,
                        input_ids=None,
                        attention_mask=None,
                        token_type_ids=None,
                        position_ids=None,
                        head_mask=None,
                        inputs_embeds=None,
                        labels=None,
                        output_attentions=None,
                        output_hidden_states=None,
                        return_dict=None):
        # 1.使用bert进行编码
        outputs = self.bert(input_ids, attention_mask=attention_mask, return_dict=True)
        # 2.取cls的表示
        cls_output = outputs.last_hidden_state[:, 0]
        # 3.使用MLP进行投影
        cls_output = self.mlp(cls_output)
        # 返回
        if not return_dict:
            return (outputs[0], cls_output) + outputs[2:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            pooler_output=cls_output,
            last_hidden_state=outputs.last_hidden_state,
            hidden_states=outputs.hidden_states,
        )

    def cl_forward(self,
                   input_ids=None,
                   attention_mask=None,
                   token_type_ids=None,
                   position_ids=None,
                   head_mask=None,
                   inputs_embeds=None,
                   labels=None,
                   output_attentions=None,
                   output_hidden_states=None,
                   return_dict=None):
        # input_ids: batch_size, num_sent, len
        batch_size = input_ids.size(0)
        num_sent = input_ids.size(1)  # 2
        # 1. 重塑输入张量的形状，使其满足bert对输入的要求
        # input_ids: batch_size * num_sent, len
        input_ids = input_ids.view((-1, input_ids.size(-1)))
        attention_mask = attention_mask.view((-1, attention_mask.size(-1)))
        # 2. 使用bert进行编码
        outputs = self.bert(input_ids, attention_mask=attention_mask, return_dict=True)
        # 3. 取cls的向量表示
        cls_output = outputs.last_hidden_state[:, 0]
        # 4. 重塑形状
        cls_output = cls_output.view((batch_size, num_sent, cls_output.size(-1)))
        # 5. 全连接层投影
        # batch_size, num_sent, 768
        cls_output = self.mlp(cls_output)
        # 6. 将同一批样本的两次向量表示分开
        z1, z2 = cls_output[:, 0], cls_output[:, 1]
        # 7. 计算两两相似度，得到相似度矩阵cos_sim
        cos_sim = self.sim(z1.unsqueeze(1), z2.unsqueeze(0))
        # 8. 生成标签[0,1,...,batch_size-1]，该标签用于提高相似度句子cos_sim对角线，并降低非对角线
        labels = torch.arange(cos_sim.size(0)).long().to(self.device)
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(cos_sim, labels)

        if not return_dict:
            output = (cos_sim,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=cos_sim,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


model = BertForCL.from_pretrained(data_args.model_name_or_path)
cl_out = model(**batch, return_dict=True)
print(cl_out.keys())

odict_keys(['loss', 'logits'])

# 五、模型训练
model.resize_token_embeddings(len(tokenizer))
trainer = Trainer(model=model,
                  train_dataset=train_dataset,
                  args=training_args,
                  tokenizer=tokenizer,
                  data_collator=collate_fn)
trainer.train()
trainer.save_model("models/test")

# 原文链接：https://blog.csdn.net/bqw18744018044/article/details/119336466

def main():
    pass


if __name__ == '__main__':
    main()
