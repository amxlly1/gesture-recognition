# -*- coding: utf-8 -*-
"""
终极合并训练：
Kaggle 21600 + 原始 4320 + 采集 296 → 三合一
用 Resize 统一大小，强增强让模型泛化到真实摄像头
"""
import sys, os, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from torch import nn
from torchvision.transforms import ToTensor, Compose, Resize, RandomHorizontalFlip, RandomRotation, ColorJitter, Lambda
from torch.utils.data import DataLoader, ConcatDataset
from dataset import CustomDataset
from model import CustomNet
from train import RandomBackgroundReplace, RandomBlur

BATCH_SIZE = 32
LEARNING_RATE = 1e-3
EPOCH = 30

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] 设备: {device}")

# 重新训练
model = CustomNet(num_classes=6)
model.to(device)
print("[INFO] 新建模型")

# === 数据集1: Kaggle 21600张（黑色背景，轻增强） ===
kaggle_ds = CustomDataset('external_data/labels.txt', '.',
    lambda: Compose([
        Lambda(lambda img: img.convert('RGB')),  # 统一转RGB
        Resize((64, 64)),
        RandomRotation(degrees=15),
        RandomHorizontalFlip(p=0.5),
        ToTensor(),
    ]))

# === 数据集2: 原始训练集 4320张（强增强+背景替换） ===
original_ds = CustomDataset('./images/train.txt', './images/train',
    lambda: Compose([
        RandomBackgroundReplace(p=0.5),
        RandomRotation(degrees=25),
        RandomHorizontalFlip(p=0.5),
        ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        RandomBlur(p=0.1),
        ToTensor(),
    ]))

# === 数据集3: 你的采集数据 296张（轻增强） ===
captured_ds = CustomDataset('captured/labels.txt', '.',
    lambda: Compose([
        Lambda(lambda img: img.convert('RGB')),
        Resize((64, 64)),
        RandomHorizontalFlip(p=0.5),
        RandomRotation(degrees=10),
        ColorJitter(brightness=0.15, contrast=0.15),
        ToTensor(),
    ]))

# 合并
merged = ConcatDataset([kaggle_ds, original_ds, captured_ds])
merged_dl = DataLoader(merged, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
print(f"[INFO] 合并: {len(kaggle_ds)} + {len(original_ds)} + {len(captured_ds)} = {len(merged)} 张")

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCH)

print(f"\n[INFO] 训练 (EPOCH={EPOCH}, LR={LEARNING_RATE}, 总数据={len(merged)})\n")

best_loss = float('inf')
for epoch in range(EPOCH):
    model.train()
    total_loss = 0.0
    correct = 0
    total_samples = 0

    for batch_data in merged_dl:
        X = batch_data['image'].to(device)
        y = batch_data['label'].to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (pred.argmax(dim=1) == y).sum().item()
        total_samples += len(y)

    scheduler.step()
    avg_loss = total_loss / len(merged_dl)
    acc = correct / total_samples * 100
    print(f"Epoch [{epoch+1:>3d}/{EPOCH}]  Loss: {avg_loss:.4f}  Acc: {acc:.2f}%")

    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model, './models/model_final.pkl')

print(f"\n[INFO] 模型已保存: ./models/model_final.pkl")

# 测试集验证
print("\n--- 原始测试集验证 ---")
model.eval()
test_ds = CustomDataset('./images/test.txt', './images/test', lambda: Compose([Lambda(lambda img: img.convert('RGB')), ToTensor()]))
test_dl = DataLoader(test_ds, batch_size=64)
correct = 0
with torch.no_grad():
    for batch_data in test_dl:
        X = batch_data['image'].to(device)
        y = batch_data['label'].to(device)
        correct += (model(X).argmax(dim=1) == y).sum().item()
print(f"原始测试集: {correct}/{len(test_ds)} = {100*correct/len(test_ds):.2f}%")
