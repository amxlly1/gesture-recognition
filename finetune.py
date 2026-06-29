# -*- coding: utf-8 -*-
"""
微调训练脚本
基于已训练好的模型，用摄像头采集的真实手势数据微调
使模型适应真实环境下的手势特征
"""

import sys
import os
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from torch import nn
from torchvision.transforms import ToTensor, Compose, RandomHorizontalFlip, RandomRotation, ColorJitter
from torch.utils.data import DataLoader, Subset
from dataset import CustomDataset
from model import CustomNet

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BATCH_SIZE = 8              # 采集数据量少，用小batch
LEARNING_RATE = 1e-4        # 微调用小学习率
EPOCH = 30

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] 设备: {device}")

# ---- 1. 加载预训练模型 ----
model = torch.load('./models/model.pkl', map_location=device, weights_only=False)
model.to(device)
print("[INFO] 预训练模型已加载")

# ---- 2. 加载采集数据 ----
captured_labels = os.path.join('captured', 'labels.txt')
if not os.path.exists(captured_labels):
    print("[ERROR] captured/labels.txt 不存在！请先运行 capture_data.py 采集数据。")
    sys.exit(1)

# 简单标签转换（CustomDataset 的 transform 参数期望可调用）
captured_ds = CustomDataset(captured_labels, '.',
    lambda: Compose([
        RandomHorizontalFlip(p=0.5),
        RandomRotation(degrees=10),
        ColorJitter(brightness=0.15, contrast=0.15),
        ToTensor(),
    ]))
captured_dl = DataLoader(captured_ds, batch_size=BATCH_SIZE, shuffle=True)
print(f"[INFO] 采集数据: {len(captured_ds)} 张")

# ---- 3. 同时加载原始训练集小部分（防止灾难性遗忘） ----
original_ds = CustomDataset('./images/train.txt', './images/train', ToTensor)
# 只用 30% 原始数据
subset_size = len(original_ds) // 3
indices = random.sample(range(len(original_ds)), subset_size)
original_subset = Subset(original_ds, indices)
original_dl = DataLoader(original_subset, batch_size=BATCH_SIZE, shuffle=True)
print(f"[INFO] 原始训练集（防遗忘）: {len(original_subset)} 张")

# ---- 4. 训练 ----
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCH)

print(f"\n[INFO] 开始微调训练 (EPOCH={EPOCH}, LR={LEARNING_RATE})\n")

for epoch in range(EPOCH):
    model.train()
    total_loss = 0.0
    correct = 0
    total_batches = 0

    # 交替使用采集数据和原始数据
    original_iter = iter(original_dl)

    for batch_data in captured_dl:
        # 采集数据
        X_c = batch_data['image'].to(device)
        y_c = batch_data['label'].to(device)
        pred_c = model(X_c)
        loss_c = loss_fn(pred_c, y_c)

        # 原始数据（防遗忘）——每个采集batch配一个原始batch
        try:
            batch_orig = next(original_iter)
        except StopIteration:
            original_iter = iter(original_dl)
            batch_orig = next(original_iter)

        X_o = batch_orig['image'].to(device)
        y_o = batch_orig['label'].to(device)
        pred_o = model(X_o)
        loss_o = loss_fn(pred_o, y_o)

        # 加权合并：采集数据权重更高
        loss = loss_c * 0.7 + loss_o * 0.3

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (pred_c.argmax(dim=1) == y_c).sum().item()
        total_batches += 1

    scheduler.step()

    avg_loss = total_loss / max(total_batches, 1)
    acc = correct / len(captured_ds) * 100
    print(f"Epoch [{epoch+1:>3d}/{EPOCH}]  Loss: {avg_loss:.4f}  "
          f"采集集准确率: {acc:.2f}%  ({correct}/{len(captured_ds)})")

# ---- 5. 保存微调后的模型 ----
torch.save(model, './models/model_finetuned.pkl')
print(f"\n[INFO] 微调模型已保存: ./models/model_finetuned.pkl")
