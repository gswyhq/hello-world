# !/usr/bin/env python3
"""

使用ChatGPT中Reward Model的思路训练一个RM，因为是打分模型，所以使用BERT（而非GPT）模型训练。
基于人工排序训练奖励模型_reward_model.py
来源：https://github.com/HarderThenHarder/transformers_tasks/blob/main/RLHF/train_reward_model.py

"""
import os
import time
import traceback
import argparse
from functools import partial

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, default_data_collator, get_scheduler


def convert_example(examples: dict, tokenizer, max_seq_len: int):
    """
    将样本数据转换为模型接收的输入数据。

    Args:
        examples (dict): 训练数据样本, e.g. -> {
                                                "text": [
                                                            '句子1	句子2	句子3',
                                                            '句子1	句子2	句子3',
                                                            ...
                                                ]
                                            }

    Returns:
        dict (str: np.array) -> tokenized_output = {
                            'input_ids': [
                                            [[101, 3928, ...], [101, 4395, ...], [101, 2135, ...]],
                                            [[101, 3928, ...], [101, 4395, ...], [101, 2135, ...]],
                                            ...
                                        ],
                            'token_type_ids': [
                                                [[0, 0, ...], [0, 0, ...], [0, 0, ...]],
                                                [[0, 0, ...], [0, 0, ...], [0, 0, ...]],
                                                ...
                                            ]
                            'position_ids': [
                                                [[0, 1, 2, ...], [0, 1, 2, ...], [0, 1, 2, ...]],
                                                [[0, 1, 2, ...], [0, 1, 2, ...], [0, 1, 2, ...]],
                                                ...
                                            ]
                            'attention_mask': [
                                                [[1, 1, ...], [1, 1, ...], [1, 1, ...]],
                                                [[1, 1, ...], [1, 1, ...], [1, 1, ...]],
                                                ...
                                            ]
                        }
    """
    tokenized_output = {
        'input_ids': [],
        'token_type_ids': [],
        'position_ids': [],
        'attention_mask': []
    }

    for example in examples['text']:
        try:
            rank_texts = example.strip().split('\t')
        except:
            print(f'"{example}" -> {traceback.format_exc()}')
            exit()

        rank_texts_prop = {
            'input_ids': [],
            'token_type_ids': [],
            'position_ids': [],
            'attention_mask': []
        }
        for rank_text in rank_texts:
            encoded_inputs = tokenizer(
                text=rank_text,
                truncation=True,
                max_length=max_seq_len,
                padding='max_length')
            rank_texts_prop['input_ids'].append(encoded_inputs["input_ids"])
            rank_texts_prop['token_type_ids'].append(encoded_inputs["token_type_ids"])
            rank_texts_prop['position_ids'].append([i for i in range(len(encoded_inputs["input_ids"]))])
            rank_texts_prop['attention_mask'].append(encoded_inputs["attention_mask"])

        for k, v in rank_texts_prop.items():
            tokenized_output[k].append(v)

    for k, v in tokenized_output.items():
        tokenized_output[k] = np.array(v)

    return tokenized_output


class RewardModel(nn.Module):

    def __init__(self, encoder):
        """
        init func.

        Args:
            encoder (transformers.AutoModel): backbone, 默认使用 ernie 3.0
        """
        super().__init__()
        self.encoder = encoder
        self.reward_layer = nn.Linear(768, 1)

    def forward(
            self,
            input_ids: torch.tensor,
            token_type_ids: torch.tensor,
            attention_mask=None,
            pos_ids=None,
    ) -> torch.tensor:
        """
        forward 函数，返回每句话的得分值。

        Args:
            input_ids (torch.tensor): (batch, seq_len)
            token_type_ids (torch.tensor): (batch, seq_len)
            attention_mask (torch.tensor): (batch, seq_len)
            pos_ids (torch.tensor): (batch, seq_len)

        Returns:
            reward: (batch, 1)
        """
        pooler_output = self.encoder(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=pos_ids,
            attention_mask=attention_mask,
        )["pooler_output"]  # (batch, hidden_size)
        reward = self.reward_layer(pooler_output)  # (batch, 1)
        return reward


def compute_rank_list_loss(rank_rewards_list: List[List[torch.tensor]], device='cpu') -> torch.Tensor:
    """
    通过给定的有序（从高到低）的ranklist的reward列表，计算rank loss。
    所有排序高的句子的得分减去排序低的句子的得分差的总和，并取负。

    Args:
        rank_rewards_list (torch.tensor): 有序（从高到低）排序句子的reward列表，e.g. ->
                                        [
                                            [torch.tensor([0.3588]), torch.tensor([0.2481]), ...],
                                            [torch.tensor([0.5343]), torch.tensor([0.2442]), ...],
                                            ...
                                        ]
        device (str): 使用设备

    Returns:
        loss (torch.tensor): tensor([0.4891], grad_fn=<DivBackward0>)
    """
    if type(rank_rewards_list) != list:
        raise TypeError(f'@param rank_rewards expected "list", received {type(rank_rewards)}.')

    loss, add_count = torch.tensor([0]).to(device), 0
    for rank_rewards in rank_rewards_list:
        for i in range(len(rank_rewards) - 1):  # 遍历所有前项-后项的得分差
            for j in range(i + 1, len(rank_rewards)):
                diff = F.logsigmoid(rank_rewards[i] - rank_rewards[j])  # sigmoid到0~1之间
                loss = loss + diff
                add_count += 1
    loss = loss / add_count
    return -loss  # 要最大化分差，所以要取负数


parser = argparse.ArgumentParser()
parser.add_argument("--model", default='bert-base-chinese', type=str, help="backbone of encoder.")
parser.add_argument("--train_path", default=None, type=str, help="The path of train set.")
parser.add_argument("--dev_path", default=None, type=str, help="The path of dev set.")
parser.add_argument("--save_dir", default="./checkpoints", type=str, required=False, help="The output directory where the model predictions and checkpoints will be written.")
parser.add_argument("--max_seq_len", default=512, type=int,help="The maximum total input sequence length after tokenization. Sequences longer "
    "than this will be truncated, sequences shorter will be padded.", )
parser.add_argument("--batch_size", default=16, type=int, help="Batch size per GPU/CPU for training.", )
parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
parser.add_argument("--num_train_epochs", default=10, type=int, help="Total number of training epochs to perform.")
parser.add_argument("--warmup_ratio", default=0.0, type=float, help="Linear warmup over warmup_ratio * total_steps.")
parser.add_argument("--valid_steps", default=200, type=int, required=False, help="evaluate frequecny.")
parser.add_argument("--logging_steps", default=10, type=int, help="log interval.")
parser.add_argument("--img_log_dir", default='logs', type=str, help="Logging image path.")
parser.add_argument("--img_log_name", default='Model Performance', type=str, help="Logging image file name.")
parser.add_argument('--device', default="cuda:0", help="Select which device to train model, defaults to gpu.")
args = parser.parse_args()



def evaluate_model(model, data_loader):
    """
    在测试集上评估当前模型的训练效果。

    Args:
        model: 当前模型
        data_loader: 测试集的dataloader
    """
    model.eval()
    with torch.no_grad():
        batch_rank_rewards = []
        for batch in data_loader:
            for batch_idx in range(len(batch['input_ids'])):
                rank_texts_count = len(batch['input_ids'][batch_idx])
                rank_rewards = []
                for text_idx in range(rank_texts_count):
                    reward = model(
                        batch['input_ids'][batch_idx][text_idx].unsqueeze(dim=0).to(args.device),
                        batch['token_type_ids'][batch_idx][text_idx].unsqueeze(dim=0).to(args.device),
                        batch['attention_mask'][batch_idx][text_idx].unsqueeze(dim=0).to(args.device),
                        batch['position_ids'][batch_idx][text_idx].unsqueeze(dim=0).to(args.device),
                    )
                    rank_rewards.append(reward[0])                      # (rank_text_num) -> [tensor([0.1696]), tensor([0.3466])]
                batch_rank_rewards.append(rank_rewards)                 # (batch, rank_text_num) -> [[tensor([0.1696]), tensor([0.3466])], ...]
    model.train()
    total_ranklist, right_ranklist = 0, 0
    for rank_rewards in batch_rank_rewards:
        rank_rewards = [t.cpu().float() for t in rank_rewards]
        rank_rewards_sorted = sorted(rank_rewards, reverse=True)
        total_ranklist += 1
        if rank_rewards_sorted == rank_rewards:
            right_ranklist += 1
    return right_ranklist / total_ranklist


def train():
    encoder = AutoModel.from_pretrained(args.model)
    model = RewardModel(encoder=encoder)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dataset = load_dataset('text', data_files={'train': args.train_path,
                                                'dev': args.dev_path})    
    print(dataset)
    convert_func = partial(convert_example, tokenizer=tokenizer, max_seq_len=args.max_seq_len)
    dataset = dataset.map(convert_func, batched=True)
    
    train_dataset = dataset["train"]
    eval_dataset = dataset["dev"]
    train_dataloader = DataLoader(train_dataset, shuffle=False, collate_fn=default_data_collator, batch_size=args.batch_size)
    eval_dataloader = DataLoader(eval_dataset, shuffle=False, collate_fn=default_data_collator, batch_size=args.batch_size)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    model.to(args.device)

    # 根据训练轮数计算最大训练步数，以便于scheduler动态调整lr
    num_update_steps_per_epoch = len(train_dataloader)
    max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    warm_steps = int(args.warmup_ratio * max_train_steps)
    lr_scheduler = get_scheduler(
        name='linear',
        optimizer=optimizer,
        num_warmup_steps=warm_steps,
        num_training_steps=max_train_steps,
    )

    loss_list = []
    tic_train = time.time()
    global_step, best_acc = 0, 0
    for epoch in range(1, args.num_train_epochs+1):
        for batch in train_dataloader:
            batch_rank_rewards = []
            for batch_idx in range(len(batch['input_ids'])):
                rank_texts_count = len(batch['input_ids'][batch_idx])
                rank_rewards = []
                for text_idx in range(rank_texts_count):
                    reward = model(
                        batch['input_ids'][batch_idx][text_idx].unsqueeze(dim=0).to(args.device),
                        batch['token_type_ids'][batch_idx][text_idx].unsqueeze(dim=0).to(args.device),
                        batch['attention_mask'][batch_idx][text_idx].unsqueeze(dim=0).to(args.device),
                        batch['position_ids'][batch_idx][text_idx].unsqueeze(dim=0).to(args.device),
                    )
                    rank_rewards.append(reward[0])                      # (rank_text_num) -> [tensor([0.1696]), tensor([0.3466])]
                batch_rank_rewards.append(rank_rewards)                 # (batch, rank_text_num) -> [[tensor([0.1696]), tensor([0.3466])], ...]
            loss = compute_rank_list_loss(batch_rank_rewards, device=args.device)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            loss_list.append(float(loss.cpu().detach()))
            
            global_step += 1
            if global_step % args.logging_steps == 0:
                time_diff = time.time() - tic_train
                loss_avg = sum(loss_list) / len(loss_list)
                print('train/train_loss', loss_avg, global_step)
                print("global step %d, epoch: %d, loss: %.5f, speed: %.2f step/s"
                        % (global_step, epoch, loss_avg, args.logging_steps / time_diff))
                tic_train = time.time()

            if global_step % args.valid_steps == 0:
                cur_save_dir = os.path.join(args.save_dir, "model_%d" % global_step)
                if not os.path.exists(cur_save_dir):
                    os.makedirs(cur_save_dir)
                torch.save(model, os.path.join(cur_save_dir, 'model.pt'))
                tokenizer.save_pretrained(cur_save_dir)
                acc = evaluate_model(model, eval_dataloader)
                print('eval/accuracy', acc, global_step)

                print("Evaluation acc: %.5f" % (acc))
                if acc > best_acc:
                    print(
                        f"best F1 performence has been updated: {best_acc:.5f} --> {acc:.5f}"
                    )
                    best_acc = acc
                    cur_save_dir = os.path.join(args.save_dir, "model_best")
                    if not os.path.exists(cur_save_dir):
                        os.makedirs(cur_save_dir)
                    torch.save(model, os.path.join(cur_save_dir, 'model.pt'))
                    tokenizer.save_pretrained(cur_save_dir)
                tic_train = time.time()


if __name__ == '__main__':
    from rich import print
    train()

# python train_reward_model.py \
#     --model "nghuyong/ernie-3.0-base-zh" \
#     --train_path "data/reward_datasets/sentiment_analysis/train.tsv" \
#     --dev_path "data/reward_datasets/sentiment_analysis/dev.tsv" \
#     --save_dir "checkpoints/reward_model/sentiment_analysis" \
#     --img_log_dir "logs/reward_model/sentiment_analysis" \
#     --img_log_name "ERNIE Reward Model" \
#     --batch_size 32 \
#     --max_seq_len 128 \
#     --learning_rate 1e-5 \
#     --valid_steps 50 \
#     --logging_steps 10 \
#     --num_train_epochs 10 \
#     --device "cuda:0"