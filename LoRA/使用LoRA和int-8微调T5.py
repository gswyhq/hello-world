#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 资料来源：https://github.com/huggingface-cn/translation/blob/main/philschmid/2023-03-23-fine-tune-flan-t5-peft.ipynb

import os
USERNAME = os.getenv("USERNAME")

from datasets import load_dataset

# https://huggingface.co/datasets/samsum
dataset_path = rf"D:\Users\{USERNAME}\data\samsum\data"
dataset = load_dataset(dataset_path)

print(f"Train dataset size: {len(dataset['train'])}")
print(f"Test dataset size: {len(dataset['test'])}")

# Train dataset size: 14732
# Test dataset size: 819

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# 模型来源：https://huggingface.co/google/flan-t5-small
model_id = rf"D:\Users\{USERNAME}\data\flan-t5-small"

# Load tokenizer of FLAN-t5-XL
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 在开始训练之前，我们还需要对数据进行预处理。生成式文本摘要属于文本生成任务。我们将文本输入给模型，模型会输出摘要。我们需要了解输入和输出文本的长度信息，以利于我们高效地批量处理这些数据。

from datasets import concatenate_datasets
import numpy as np
# The maximum total input sequence length after tokenization.
# Sequences longer than this will be truncated, sequences shorter will be padded.
tokenized_inputs = concatenate_datasets([dataset["train"], dataset["test"]]).map(lambda x: tokenizer(x["dialogue"], truncation=True), batched=True, remove_columns=["dialogue", "summary"])
input_lenghts = [len(x) for x in tokenized_inputs["input_ids"]]
# take 85 percentile of max length for better utilization
max_source_length = int(np.percentile(input_lenghts, 85))
print(f"Max source length: {max_source_length}")
# Max source length: 255

# The maximum total sequence length for target text after tokenization.
# Sequences longer than this will be truncated, sequences shorter will be padded."
tokenized_targets = concatenate_datasets([dataset["train"], dataset["test"]]).map(lambda x: tokenizer(x["summary"], truncation=True), batched=True, remove_columns=["dialogue", "summary"])
target_lenghts = [len(x) for x in tokenized_targets["input_ids"]]
# take 90 percentile of max length for better utilization
max_target_length = int(np.percentile(target_lenghts, 90))
print(f"Max target length: {max_target_length}")
# Max target length: 50

# 我们将在训练前统一对数据集进行预处理并将预处理后的数据集保存到磁盘。你可以在本地机器或 CPU 上运行此步骤并将其上传到 Hugging Face Hub。

def preprocess_function(sample,padding="max_length"):
    # add prefix to the input for t5
    inputs = ["summarize: " + item for item in sample["dialogue"]]

    # tokenize inputs
    model_inputs = tokenizer(inputs, max_length=max_source_length, padding=padding, truncation=True)

    # Tokenize targets with the `text_target` keyword argument
    labels = tokenizer(text_target=sample["summary"], max_length=max_target_length, padding=padding, truncation=True)

    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    if padding == "max_length":
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=["dialogue", "summary", "id"])
print(f"Keys of tokenized dataset: {list(tokenized_dataset['train'].features)}")
# Keys of tokenized dataset: ['input_ids', 'attention_mask', 'labels']

# save datasets to disk for later easy loading
tokenized_dataset["train"].save_to_disk(f"{dataset_path}/train")
tokenized_dataset["test"].save_to_disk(f"{dataset_path}/eval")

# 使用 LoRA 和 bnb int-8 微调 T5
# 除了 LoRA 技术，我们还使用 bitsanbytes LLM.int8() 把冻结的 LLM 量化为 int8。这使我们能够将 FLAN-T5 XXL 所需的内存降低到约四分之一。
#
# 训练的第一步是加载模型。
import torch
from transformers import AutoModelForSeq2SeqLM
from transformers.utils.quantization_config import BitsAndBytesConfig
# load model from the hub
model = AutoModelForSeq2SeqLM.from_pretrained(model_id, load_in_8bit=True, torch_dtype=torch.float16, device_map="auto", quantization_config = BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=True))

# 我们可以使用 peft 为 LoRA int-8 训练作准备了。

from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType

# Define LoRA Config
lora_config = LoraConfig(
 r=16,
 lora_alpha=32,
 target_modules=["q", "v"],
 lora_dropout=0.05,
 bias="none",
 task_type=TaskType.SEQ_2_SEQ_LM
)
# prepare int-8 model for training
model = prepare_model_for_int8_training(model)

# add LoRA adaptor
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# trainable params: 688128 || all params: 77649280 || trainable%: 0.8862001038515747

# 如你所见，这里我们只训练了模型参数的 0.88%！这个巨大的内存增益让我们安心地微调模型，而不用担心内存问题。
# 接下来需要创建一个 DataCollator，负责对输入和标签进行填充，我们使用 🤗 Transformers 库中的DataCollatorForSeq2Seq 来完成这一环节。
from transformers import DataCollatorForSeq2Seq

# we want to ignore tokenizer pad token in the loss
label_pad_token_id = -100
# Data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=label_pad_token_id,
    pad_to_multiple_of=8
)

# 最后一步是定义训练超参 (TrainingArguments)。

from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
output_dir = rf"D:\Users\{USERNAME}\data\lora-flan-t5-small"

# Define training args
training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
		auto_find_batch_size=True,
    learning_rate=1e-3, # higher learning rate
    num_train_epochs=5,
    logging_dir=f"{output_dir}/logs",
    logging_strategy="steps",
    logging_steps=500,
    save_strategy="no",
    report_to="tensorboard",
)

# 从磁盘上读取已处理好的数据
tokenized_dataset = {"train": datasets.load_from_disk((f"{dataset_path}/train")),
                     "test": datasets.load_from_disk((f"{dataset_path}/eval"))}

# Create Trainer instance
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset["train"],
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!


# 运行下面的代码，开始训练模型。请注意，对于 T5，出于收敛稳定性考量，某些层我们仍保持 float32 精度。

# train model
trainer.train()
# windows下CPU训练报错：
# "bernoulli_scalar_cpu_" not implemented for 'Half'
# 原因：CPU环境不支持torch.float16

# 我们可以将模型保存下来以用于后面的推理和评估。我们暂时将其保存到磁盘，但你也可以使用 model.push_to_hub 方法将其上传到 Hugging Face Hub。

# Save our LoRA model & tokenizer results
peft_model_id="results"
trainer.model.save_pretrained(peft_model_id)
tokenizer.save_pretrained(peft_model_id)
# if you want to save the base model to call
# trainer.model.base_model.save_pretrained(peft_model_id)
# 最后生成的 LoRA checkpoint 文件很小，仅需 84MB 就包含了从 samsum 数据集上学到的所有知识。

# 使用 LoRA FLAN-T5 进行评估和推理
# 我们将使用 evaluate 库来评估 rogue 分数。我们可以使用 PEFT 和 transformers 来对 FLAN-T5 XXL 模型进行推理。对 FLAN-T5 XXL 模型，我们至少需要 18GB 的 GPU 显存。

import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load peft config for pre-trained checkpoint etc. 
peft_model_id = "results"
config = PeftConfig.from_pretrained(peft_model_id)

# load base LLM model and tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path,  load_in_8bit=True,  device_map={"":0})
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

# Load the Lora model
model = PeftModel.from_pretrained(model, peft_model_id, device_map={"":0})
model.eval()

print("Peft model loaded")
# 我们用测试数据集中的一个随机样本来试试摘要效果。

from datasets import load_dataset 
from random import randrange


# Load dataset from the hub and get a sample
dataset = load_dataset(dataset_path)
sample = dataset['test'][randrange(len(dataset["test"]))]

input_ids = tokenizer(sample["dialogue"], return_tensors="pt", truncation=True).input_ids.cuda()
# with torch.inference_mode():
outputs = model.generate(input_ids=input_ids, max_new_tokens=10, do_sample=True, top_p=0.9)
print(f"input sentence: {sample['dialogue']}\n{'---'* 20}")

print(f"summary:\n{tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0]}")
# 不错！我们的模型有效！现在，让我们仔细看看，并使用 test 集中的全部数据对其进行评估。
# 为此，我们需要实现一些工具函数来帮助生成摘要并将其与相应的参考摘要组合到一起。评估摘要任务最常用的指标是 rogue_score，它的全称是 Recall-Oriented Understudy for Gisting Evaluation。
# 与常用的准确率指标不同，它将生成的摘要与一组参考摘要进行比较。

import evaluate
import numpy as np
from datasets import load_from_disk
from tqdm import tqdm

# Metric
metric = evaluate.load("rouge")

def evaluate_peft_model(sample,max_target_length=50):
    # generate summary
    outputs = model.generate(input_ids=sample["input_ids"].unsqueeze(0).cuda(), do_sample=True, top_p=0.9, max_new_tokens=max_target_length)    
    prediction = tokenizer.decode(outputs[0].detach().cpu().numpy(), skip_special_tokens=True)
    # decode eval sample
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(sample['labels'] != -100, sample['labels'], tokenizer.pad_token_id)
    labels = tokenizer.decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    return prediction, labels

# load test dataset from distk
test_dataset = load_from_disk(f"{dataset_path}/eval").with_format("torch")

# run predictions
# this can take ~45 minutes
predictions, references = [] , []
for sample in tqdm(test_dataset):
    p,l = evaluate_peft_model(sample)
    predictions.append(p)
    references.append(l)

# compute metric 
rogue = metric.compute(predictions=predictions, references=references, use_stemmer=True)

# print results 
print(f"Rogue1: {rogue['rouge1']* 100:2f}%")
print(f"rouge2: {rogue['rouge2']* 100:2f}%")
print(f"rougeL: {rogue['rougeL']* 100:2f}%")
print(f"rougeLsum: {rogue['rougeLsum']* 100:2f}%")

# Rogue1: 50.386161%
# rouge2: 24.842412%
# rougeL: 41.370130%
# rougeLsum: 41.394230%
# 我们 PEFT 微调后的 FLAN-T5-XXL 在测试集上取得了 50.38% 的 rogue1 分数。相比之下，flan-t5-base 的全模型微调获得了 47.23 的 rouge1 分数。rouge1 分数提高了 3% 。

# 令人难以置信的是，我们的 LoRA checkpoint 只有 84MB，而且性能比对更小的模型进行全模型微调后的 checkpoint 更好。

def main():
    pass


if __name__ == '__main__':
    main()
