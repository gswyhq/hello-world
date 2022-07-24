#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# pip install pytorch-lightning

import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl

# 第 2 步：定义 LightningModule（nn.Module 子类）
# LightningModule 定义了一个完整的系统（即：GAN、自动编码器、BERT 或简单的图像分类器）。

class LitAutoEncoder(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(28 * 28, 128), nn.ReLU(), nn.Linear(128, 3))
        self.decoder = nn.Sequential(nn.Linear(3, 128), nn.ReLU(), nn.Linear(128, 28 * 28))

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        embedding = self.encoder(x)
        return embedding

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
# 注意： Training_step 定义了训练循环。Forward 定义 LightningModule 在推理/预测期间的行为方式。

# Step 3: 训练!
dataset = MNIST(os.getcwd(), download=True, transform=transforms.ToTensor())
train, val = random_split(dataset, [55000, 5000])

autoencoder = LitAutoEncoder()
trainer = pl.Trainer()
trainer.fit(autoencoder, DataLoader(train), DataLoader(val))

def main():
    pass


if __name__ == '__main__':
    main()