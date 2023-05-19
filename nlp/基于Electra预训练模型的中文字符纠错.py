#!/usr/lib/python3
# -*- coding: utf-8 -*-


import torch

# from easytokenizer import AutoTokenizer
from transformers import AutoTokenizer
# from model import ElectraForSpellingCheck
from transformers import ElectraModel
from transformers import pipelines
import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForMaskedLM

# https://huggingface.co/bert-base-chinese
mask_model_dir = "/home/gswyhq/huggingface/bert-base-chinese"
mask_tokenizer = AutoTokenizer.from_pretrained(mask_model_dir)
mask_model = AutoModelForMaskedLM.from_pretrained(mask_model_dir)

# https://huggingface.co/WangZeJun/electra-base-chinese-spelling-check/tree/main
model_file = "/home/gswyhq/huggingface/electra-base-chinese-spelling-check/pytorch_model.bin"
model_dir = "/home/gswyhq/huggingface/electra-base-chinese-spelling-check"

# https://huggingface.co/hfl/chinese-electra-180g-base-discriminator/tree/main
pretrained_model_name_or_path = "/home/gswyhq/huggingface/chinese-electra-180g-base-discriminator"



# from loss import BinaryLabelSmoothLoss, BinaryFocalLoss
# transformers.__version__
# Out[3]: '4.27.4'

# 来源： https://github.com/gged/ElectraForSpellingCheck

def binary_focal_loss(input, target, alpha=None, gamma=2, reduction="mean", pos_weight=None):
    # Compute focal loss for binary classification
    p = input.sigmoid()
    factor = ((1 - p) * target + p * (1 - target)).pow(gamma)
    if alpha is not None:
        factor = (alpha * target + (1 - alpha) * (1 - target)) * factor
    if pos_weight is not None:
        pos_weight = torch.tensor(pos_weight)
    loss = F.binary_cross_entropy(p, target, reduction="none", pos_weight=pos_weight) * factor
    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    return loss


def binary_label_smooth_loss(input, target, label_smoothing=0.1, reduction="mean", pos_weight=None):
    # Compute label smooth loss for binary classification
    if label_smoothing:
        target = target * (1 - label_smoothing) + 0.5 * label_smoothing
    if pos_weight is not None:
        pos_weight = torch.tensor(pos_weight)
    return F.binary_cross_entropy_with_logits(input, target, reduction=reduction, pos_weight=pos_weight)


class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction="mean", pos_weight=None):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.pos_weight = torch.tensor(pos_weight) if pos_weight is not None else None

    def forward(self, input, target):
        p = input.sigmoid()
        factor = ((1 - p) * target + p * (1 - target)).pow(self.gamma)
        if self.alpha is not None:
            factor = (self.alpha * target + (1 - self.alpha) * (1 - target)) * factor
        loss = F.binary_cross_entropy(p, target, reduction="none", pos_weight=self.pos_weight) * factor
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return loss


class BinaryLabelSmoothLoss(nn.Module):
    def __init__(self, label_smoothing=0.1, reduction="mean", pos_weight=None):
        super(BinaryLabelSmoothLoss, self).__init__()
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        self.pos_weight = torch.tensor(pos_weight) if pos_weight is not None else None

    def forward(self, input, target):
        if self.label_smoothing:
            target = target * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        return F.binary_cross_entropy_with_logits(input, target, reduction=self.reduction,
                                                  pos_weight=self.pos_weight)

class ElectraForSpellingCheck(nn.Module):
    def __init__(self, pretrained_model):
        super(ElectraForSpellingCheck, self).__init__()

        self.hidden_size = pretrained_model.config.hidden_size
        self.pad_token_id = pretrained_model.config.pad_token_id

        self.electra = pretrained_model
        self.detector = nn.Linear(self.hidden_size, 1)

    def forward(
            self,
            input_ids,
            attention_mask=None,
            token_type_ids=None,
            labels=None,
            loss_type="bce",
            reduction="mean",
            alpha=None,
            gamma=None,
            label_smoothing=None,
            pos_weight=None
    ):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
            attention_mask[input_ids == self.pad_token_id] = 0

        hidden_states = self.electra(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=False
        )

        sequence_output = hidden_states[0]
        logits = self.detector(sequence_output).squeeze(-1)

        loss = None
        if labels is not None:
            if loss_type == "bce":
                if pos_weight is not None:
                    pos_weight = torch.tensor(pos_weight)
                loss_fct = nn.BCEWithLogitsLoss(
                    reduction=reduction,
                    pos_weight=pos_weight
                )
            elif loss_type == "lsr":
                loss_fct = BinaryLabelSmoothLoss(
                    label_smoothing=label_smoothing,
                    reduction=reduction,
                    pos_weight=pos_weight
                )
            elif loss_type == "focal":
                loss_fct = BinaryFocalLoss(
                    alpha=alpha,
                    gamma=gamma,
                    reduction=reduction,
                    pos_weight=pos_weight
                )
            else:
                raise ValueError("Unsupported loss function type!")

            active_loss = attention_mask.view(-1, sequence_output.shape[1]) == 1
            active_logits = logits.view(-1, sequence_output.shape[1])[active_loss]
            active_labels = labels[active_loss]
            loss = loss_fct(active_logits, active_labels.float())

        return (loss, logits) if loss is not None else logits

def detect(device, model, tokenizer, text, max_length=256):
    # encoding = tokenizer.encode(text, truncation=True, max_length=max_seq_length)
    encoding = tokenizer.encode_plus(text, truncation=True, max_length=max_length, return_offsets_mapping=True)
    input_ids = torch.tensor([encoding["input_ids"]], device=device)
    offsets = encoding["offset_mapping"]
    with torch.no_grad():
        logits = model(input_ids)
    predictions = logits.squeeze(dim=0).sigmoid().round()
    index = torch.nonzero(predictions).squeeze(dim=-1).tolist()
    output = []
    for idx in index:
        idx = offsets[idx]
        output.append((idx, text[idx[0]: idx[1]]))
    print(text)
    print("检查结果: ", output)
    print('-'*40)
    return output

def fill_mask(mask_model, mask_tokenizer, text='北京是[MASK]国首都'):
    '''对字符串中的mask指进行预测'''
    if mask_tokenizer.mask_token not in text:
        return text
    fill = pipelines.pipeline('fill-mask', model=mask_model, tokenizer=mask_tokenizer)
    output = fill(text, top_k=1)
    if isinstance(output[0], list):
        for ret in output:
            token_str = ret[0]['token_str']
            text = text.replace(mask_tokenizer.mask_token, token_str, 1)
        return text
    else:
        return output[0]['sequence'].replace(' ', '')


def main():
    # tokenizer = AutoTokenizer(args.vocab_file, do_lower_case=args.do_lower_case)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    pretrained_model = ElectraModel.from_pretrained(pretrained_model_name_or_path)
    model = ElectraForSpellingCheck(pretrained_model)
    state_dict = torch.load(model_file, map_location=device)
    model.load_state_dict(state_dict)
    del state_dict
    model.to(device)
    model.eval()

    text_list = ['我的台北好友老蔡，在大陆开办奶牛养直场和葡萄园，孩子考上了在武汉的大学。',
 '祖国大陆始终扩开胸怀期待游子，相信血浓于水的亲情定能跨越浅浅的海峡。',
 '从党的二十大报告，到中央经济工作会议，再到政府工作报告，都在召示着这样一个事实：以习近平同志为核心的党中央始终坚持党对经济工作的全面领导，坚持稳中求进工作总基调，坚持实事求是、尊重规律、系统观念、底线思维，正确认识困难挑战，驾又经济工作的能力不断加强，做好经济工作的信心一以贯之。',
 '新增10个社区养老服务一站，就近为有需求的居家老年人提供生活照料、陪伴护理等多样化服务，提升老年人生活质量。',
 '这些成就是中国人民团结一心、砥厉奋进的结果,也与外国友人的关心和支持密不可分。',
 '智能控备的普遍应用，让业务办理由人工转变为客户自助与半自助，实现了操作风险的部分转移，使柜面操作风险有效降压。但从银行声誉风险角度来讲，由于客户自助操作而引起的风险，更容易引起声誉风险。']
    for text in text_list:
        outputs = detect(device, model, tokenizer, text)
        for (start, end), _ in outputs[::-1]:
            text = text[:start] + mask_tokenizer.mask_token + text[end:]
        print('纠错后的结果：', fill_mask(mask_model, mask_tokenizer, text))
        print('\n')
    while True:
        text = input("请输入待检查的文本(最长256个字符, quit/q 退出): ")
        if text in ["quit", "q"]:
            break
        outputs = detect(device, model, tokenizer, text)
        for (start, end), _ in outputs[::-1]:
            text = text[:start] + mask_tokenizer.mask_token + text[end:]
        print('纠错后的结果：', fill_mask(mask_model, mask_tokenizer, text))
        print('\n')

if __name__ == '__main__':
    main()
