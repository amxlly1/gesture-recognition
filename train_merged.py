# -*- coding: utf-8 -*-
"""
合并微调训练：原始4320张 + 你的296张 → 联合训练
保留泛化能力，同时适配你的手势特征
"""
import sys, os, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from torch import nn
from torchvision.transforms import ToTensor, Compose, RandomHorizontalFlip, RandomRotation, ColorJitter, Resize
from torch.utils.data import DataLoader, ConcatDataset
from dataset import CustomDataset
from model import CustomNet

BATCH_SIZE = 32
LEARNING_RATE = 5e-4         # 中等学习率
EPOCH = 40

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] 设备: {device}")

# 重新训练（不用旧模型，从头训练让所有数据公平参与）
model = CustomNet(num_classes=6)
model.to(device)
print("[INFO] 新建模型")

# 原始训练集（强增强：背景替换等，保持泛化能力）
from train import RandomBackgroundReplace, RandomBlur
original_ds = CustomDataset('./images/train.txt', './images/train',
    lambda: Compose([
        RandomBackgroundReplace(p=0.5),
        RandomRotation(degrees=25),
        RandomHorizontalFlip(p=0.5),
        ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        RandomBlur(p=0.1),
        ToTensor(),
    ]))

# 你的采集数据（轻增强，因为数据少，过度增强反而有害）
captured_ds = CustomDataset('captured/labels.txt', '.',
    lambda: Compose([
        Resize((64, 64)),              # 统一尺寸
        RandomHorizontalFlip(p=0.5),
        RandomRotation(degrees=10),
        ColorJitter(brightness=0.15, contrast=0.15),
        ToTensor(),
    ]))

# 合并：原始4320 + 采集296 = 4616张
merged_ds = ConcatDataset([original_ds, captured_ds])
merged_dl = DataLoader(merged_ds, batch_size=BATCH_SIZE, shuffle=True)
print(f"[INFO] 合并数据集: {len(original_ds)} + {len(captured_ds)} = {len(merged_ds)} 张")

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCH)

print(f"\n[INFO] 开始训练 (EPOCH={EPOCH}, LR={LEARNING_RATE})\n")

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
        torch.save(model, './models/model_merged.pkl')

print(f"\n[INFO] 模型已保存: ./models/model_merged.pkl")

# ---- 快速测试原测试集 ----
print("\n--- 测试集验证 ---")
model.eval()
test_ds = CustomDataset('./images/test.txt', './images/test', ToTensor)
test_dl = DataLoader(test_ds, batch_size=64)
correct = 0
with torch.no_grad():
    for batch_data in test_dl:
        X = batch_data['image'].to(device)
        y = batch_data['label'].to(device)
        pred = model(X).argmax(dim=1)
        correct += (pred == y).sum().item()
print(f"原始测试集准确率: {correct}/{len(test_ds)} = {100*correct/len(test_ds):.2f}%")
