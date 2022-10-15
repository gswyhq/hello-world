#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os,re
import torch
import numpy as np
from transformers import BertTokenizer, BertModel, BertForMaskedLM

USERNAME = os.getenv('USERNAME')

# transformers==3.0.2
BERT_BASE_CHINESE_PATH = rf'D:\Users\{USERNAME}\data\bert_base_pytorch\bert-base-chinese'
# BERT_BASE_CHINESE_PATH = rf'D:\Users\{USERNAME}\data\bert_base_pytorch\bert_wwm_pretrain'

# BERT_BASE_CHINESE_PATH = 'bert-base-chinese'

# Load pre-trained model tokenizer (vocabulary)
# tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
tokenizer = BertTokenizer.from_pretrained(BERT_BASE_CHINESE_PATH)

# Load pre-trained model (weights)
model = BertForMaskedLM.from_pretrained(BERT_BASE_CHINESE_PATH)
model.eval()

def mask_pred(text:str='[CLS] 深 圳 是 一 座 [MASK] [MASK] 的 城 市 [SEP]', masked_index:int=None):
    """
    对字符串中的首个mask进行预测
    这里是先预测一个 mask 位，预测完成后，再把已经预测的mask填好，继续预测下一个mask位，而不是多个mask位一起预测；
    多个mask 位一起预测的时候，针对候选比较多的情况，预测效果极差，如：'[CLS] [MASK] [MASK] 是 一 座 海 滨 的 城 市 [SEP]'
    :param text:
    :return:
    """

    # text = '[CLS] 深 圳 是 一 座 [MASK] [MASK] 的 城 市 [SEP]'
    tokenized_text = tokenizer.tokenize(text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

    # Create the segments tensors.
    segments_ids = [0] * len(tokenized_text)

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    if masked_index is None:
        masked_index = tokenized_text.index('[MASK]')

    # Predict all tokens
    with torch.no_grad():
        predictions = model(tokens_tensor, segments_tensors)
    # print(predictions)
    # predicted_index = torch.argmax(predictions[0][0][masked_index]).item()
    # predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
    # print("预测最有可能的结果：{}".format(predicted_token))
    top10_mask = []
    sorted, indices = torch.sort(predictions[0][0][masked_index], descending=True)

    for weight, index in zip(sorted[:10], indices[:10]):
        pred_token = tokenizer.convert_ids_to_tokens([index])[0]
        # print('{} -> {}'.format(pred_token, weight))
        top10_mask.append([pred_token, float(weight)])
    return top10_mask

def main(reverse=False):
    import re
    import copy
    beam_size = 50
    # text = '[CLS] 深 圳 是 一 座 [MASK] [MASK] 的 城 市 [SEP]'
    text = '[CLS] [MASK] [MASK] 是 一 座 海 滨 的 城 市 [SEP]'
    text = '[CLS] 欧 拉 是 一 位 [MASK] [MASK] 的 [MASK] [MASK] 家 。[SEP]'
    text = '[CLS] 是 一 种 [MASK] [MASK] 而 又 [MASK] [MASK] [MASK] 的 方 法 。[SEP]'
    tokenized_text = text.split()
    masked_indexs = [index for index, token in
                     enumerate(tokenized_text) if
                     token == '[MASK]']
    result_list = [[tokenized_text, 0]]
    if reverse:
        # 从后往前预测
        sort_masked_indexs = masked_indexs[::-1]
    else:
        # 从前往后预测
        sort_masked_indexs = masked_indexs
    for mask_index in sort_masked_indexs:
        tmp_result_list = copy.deepcopy(result_list)
        result_list = []
        for tmp_tokenized_text, weight in tmp_result_list:
            top10_mask = mask_pred(' '.join(tmp_tokenized_text), masked_index=mask_index)
            for token, weight in top10_mask:
                tmp_tokenized_text[mask_index] = token
                result_list.append([copy.deepcopy(tmp_tokenized_text), weight])
        result_list.sort(key=lambda x: x[1], reverse=True)
        result_list = result_list[:beam_size]

    for tokenized_text, weight in sorted(result_list, key=lambda x:x[1], reverse=True):
        print(''.join(tokenized_text), weight)

if __name__ == '__main__':
    main(reverse=False)  # reverse为真，则从后往前预测，否则从前往后预测；

# bert-base-chinese(左)和bert_wwm_pretrain(右)模型的预测结果：  (答案是：欧拉是一位著名数学家。从后往前预测时，也仅在备选答案中存在)
# [CLS]欧拉是一位真诚的艺术家。[SEP] 23.13555908203125	           [CLS]欧拉是一位有才的数学家。[SEP] 23.002662658691406
# [CLS]欧拉是一位天真的艺术家。[SEP] 23.125751495361328	           [CLS]欧拉是一位出生的数学家。[SEP] 22.938705444335938
# [CLS]欧拉是一位真挚的艺术家。[SEP] 22.941585540771484	           [CLS]欧拉是一位有钱的政治家。[SEP] 22.760164260864258
# [CLS]欧拉是一位美好的艺术家。[SEP] 22.83856964111328	           [CLS]欧拉是一位富[UNK]的政治家。[SEP] 22.690235137939453
# [CLS]欧拉是一位优雅的艺术家。[SEP] 22.74606704711914	           [CLS]欧拉是一位有为的政治家。[SEP] 22.549108505249023
# [CLS]欧拉是一位真实的艺术家。[SEP] 22.654579162597656	           [CLS]欧拉是一位有才的哲学家。[SEP] 22.39078140258789
# [CLS]欧拉是一位美丽的艺术家。[SEP] 22.605318069458008	           [CLS]欧拉是一位虔信的数学家。[SEP] 22.347013473510742
# [CLS]欧拉是一位天主的哲学家。[SEP] 22.515470504760742	           [CLS]欧拉是一位出才的数学家。[SEP] 22.345441818237305
# [CLS]欧拉是一位美貌的艺术家。[SEP] 22.437347412109375	           [CLS]欧拉是一位聪颖的数学家。[SEP] 22.32929229736328
# [CLS]欧拉是一位天真的哲学家。[SEP] 22.358121871948242	           [CLS]欧拉是一位优雅的艺术家。[SEP] 22.30482292175293
# [CLS]欧拉是一位伟大的哲学家。[SEP] 22.343036651611328	           [CLS]欧拉是一位富有的银行家。[SEP] 22.167888641357422
# [CLS]欧拉是一位优良的艺术家。[SEP] 22.276229858398438	           [CLS]欧拉是一位聪秀的数学家。[SEP] 22.121469497680664
# [CLS]欧拉是一位美艳的艺术家。[SEP] 22.26511001586914	           [CLS]欧拉是一位伟出的哲学家。[SEP] 22.10123062133789
# [CLS]欧拉是一位优质的艺术家。[SEP] 22.261808395385742	           [CLS]欧拉是一位出才的哲学家。[SEP] 22.036457061767578
# [CLS]欧拉是一位出生的哲学家。[SEP] 22.255809783935547	           [CLS]欧拉是一位重要的数学家。[SEP] 21.960783004760742
# [CLS]欧拉是一位美妙的艺术家。[SEP] 22.229387283325195	           [CLS]欧拉是一位富[UNK]的艺术家。[SEP] 21.942811965942383
# [CLS]欧拉是一位出众的艺术家。[SEP] 22.222049713134766	           [CLS]欧拉是一位有为的哲学家。[SEP] 21.908184051513672
# [CLS]欧拉是一位杰大的哲学家。[SEP] 22.162174224853516	           [CLS]欧拉是一位伟[UNK]的哲学家。[SEP] 21.89573097229004
# [CLS]欧拉是一位知性的艺术家。[SEP] 22.089988708496094	           [CLS]欧拉是一位伟大的数学家。[SEP] 21.848215103149414
# [CLS]欧拉是一位卓著的哲学家。[SEP] 22.07208251953125	           [CLS]欧拉是一位聪敏的数学家。[SEP] 21.844484329223633
# [CLS]欧拉是一位真心的艺术家。[SEP] 22.040924072265625	           [CLS]欧拉是一位杰作的雕塑家。[SEP] 21.84113121032715
# [CLS]欧拉是一位卓越的艺术家。[SEP] 22.029638290405273	           [CLS]欧拉是一位虔谨的哲学家。[SEP] 21.79823875427246
# [CLS]欧拉是一位出生的艺术家。[SEP] 22.02922248840332	           [CLS]欧拉是一位虔肃的哲学家。[SEP] 21.797903060913086
# [CLS]欧拉是一位优美的艺术家。[SEP] 22.02590560913086	           [CLS]欧拉是一位虔诚的哲学家。[SEP] 21.691415786743164
# [CLS]欧拉是一位真正的哲学家。[SEP] 22.0233211517334	           [CLS]欧拉是一位虔誠的哲學家。[SEP] 21.672595977783203
# [CLS]欧拉是一位卓尔的哲学家。[SEP] 22.0211124420166	           [CLS]欧拉是一位聪[UNK]的哲学家。[SEP] 21.669952392578125
# [CLS]欧拉是一位伟人的哲学家。[SEP] 21.995731353759766	           [CLS]欧拉是一位聪慧的数学家。[SEP] 21.626964569091797
# [CLS]欧拉是一位伟业的哲学家。[SEP] 21.971385955810547	           [CLS]欧拉是一位聪明的数学家。[SEP] 21.597944259643555
# [CLS]欧拉是一位天然的艺术家。[SEP] 21.931514739990234	           [CLS]欧拉是一位虔[UNK]的哲学家。[SEP] 21.58284568786621
# [CLS]欧拉是一位优异的艺术家。[SEP] 21.888290405273438	           [CLS]欧拉是一位聪[UNK]的数学家。[SEP] 21.54949188232422
# [CLS]欧拉是一位优越的艺术家。[SEP] 21.8848934173584	           [CLS]欧拉是一位虔好的哲学家。[SEP] 21.53812026977539
# [CLS]欧拉是一位出身的哲学家。[SEP] 21.76026153564453	           [CLS]欧拉是一位聪慧的哲学家。[SEP] 21.534847259521484
# [CLS]欧拉是一位美好的企业家。[SEP] 21.725454330444336	           [CLS]欧拉是一位虔信的哲学家。[SEP] 21.533456802368164
# [CLS]欧拉是一位天性的哲学家。[SEP] 21.652191162109375	           [CLS]欧拉是一位虔正的哲学家。[SEP] 21.525941848754883
# [CLS]欧拉是一位卓著的艺术家。[SEP] 21.6077938079834	           [CLS]欧拉是一位聪敏的哲学家。[SEP] 21.503786087036133
# [CLS]欧拉是一位知心的艺术家。[SEP] 21.52621078491211	           [CLS]欧拉是一位虔敬的哲学家。[SEP] 21.450563430786133
# [CLS]欧拉是一位真正的艺术家。[SEP] 21.524477005004883	           [CLS]欧拉是一位杰作的艺术家。[SEP] 21.438915252685547
# [CLS]欧拉是一位知性的哲学家。[SEP] 21.45025634765625	           [CLS]欧拉是一位虔睡的哲学家。[SEP] 21.40713882446289
# [CLS]欧拉是一位著作的哲学家。[SEP] 21.384069442749023	           [CLS]欧拉是一位聪颖的哲学家。[SEP] 21.376441955566406
# [CLS]欧拉是一位天生的艺术家。[SEP] 21.35021209716797	           [CLS]欧拉是一位杰作的雕刻家。[SEP] 21.331504821777344
# [CLS]欧拉是一位真理的哲学家。[SEP] 21.321056365966797	           [CLS]欧拉是一位虔誠的神學家。[SEP] 20.53133773803711
# [CLS]欧拉是一位杰作的艺术家。[SEP] 21.241249084472656	           [CLS]欧拉是一位虔诚的神学家。[SEP] 20.50921630859375
# [CLS]欧拉是一位优雅的音乐家。[SEP] 21.236358642578125	           [CLS]欧拉是一位虔谨的神学家。[SEP] 20.196767807006836
# [CLS]欧拉是一位卓尔的艺术家。[SEP] 20.872074127197266	           [CLS]欧拉是一位虔肃的神学家。[SEP] 20.10153579711914
# [CLS]欧拉是一位美妙的音乐家。[SEP] 20.5682430267334	           [CLS]欧拉是一位虔敬的神学家。[SEP] 19.95197868347168
# [CLS]欧拉是一位优美的音乐家。[SEP] 20.28182601928711	           [CLS]欧拉是一位虔[UNK]的神学家。[SEP] 19.85395050048828
# [CLS]欧拉是一位优异的科学家。[SEP] 20.110727310180664	           [CLS]欧拉是一位虔正的神学家。[SEP] 19.74327278137207
# [CLS]欧拉是一位天主的神学家。[SEP] 20.110225677490234	           [CLS]欧拉是一位虔信的神学家。[SEP] 19.69390869140625
# [CLS]欧拉是一位优美的女作家。[SEP] 17.613521575927734	           [CLS]欧拉是一位虔誠的哲学家。[SEP] 19.26153564453125
# [CLS]欧拉是一位天真的小画家。[SEP] 16.400232315063477	           [CLS]欧拉是一位虔睡的哲學家。[SEP] 19.13873291015625
