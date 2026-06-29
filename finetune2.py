# -*- coding: utf-8 -*-
import sys, os, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from torch import nn
from torchvision.transforms import ToTensor, Compose, RandomHorizontalFlip, RandomRotation, ColorJitter
from torch.utils.data import DataLoader, Subset
from dataset import CustomDataset
from model import CustomNet

BATCH_SIZE = 16
LEARNING_RATE = 1e-3          # 更高学习率
EPOCH = 60                    # 更多epoch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] 设备: {device}")

# 加载预训练模型（基于增强训练的版本）
model = torch.load('./models/model.pkl', map_location=device, weights_only=False)
model.to(device)
print("[INFO] 预训练模型已加载")

# 加载采集数据
captured_labels = os.path.join('captured', 'labels.txt')
if not os.path.exists(captured_labels):
    print("[ERROR] captured/labels.txt 不存在！")
    sys.exit(1)

captured_ds = CustomDataset(captured_labels, '.',
    lambda: Compose([
        RandomHorizontalFlip(p=0.5),
        RandomRotation(degrees=15),
        ColorJitter(brightness=0.2, contrast=0.2),
        ToTensor(),
    ]))
captured_dl = DataLoader(captured_ds, batch_size=BATCH_SIZE, shuffle=True)
print(f"[INFO] 采集数据: {len(captured_ds)} 张")

# 只用采集数据训练（不做防遗忘）
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCH)

print(f"\n[INFO] 开始训练 (EPOCH={EPOCH}, LR={LEARNING_RATE})\n")

best_acc = 0
for epoch in range(EPOCH):
    model.train()
    total_loss = 0.0
    correct = 0

    for batch_data in captured_dl:
        X = batch_data['image'].to(device)
        y = batch_data['label'].to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (pred.argmax(dim=1) == y).sum().item()

    scheduler.step()
    acc = correct / len(captured_ds) * 100
    if acc > best_acc:
        best_acc = acc
        torch.save(model, './models/model_finetuned.pkl')

    print(f"Epoch [{epoch+1:>3d}/{EPOCH}]  Loss: {total_loss/len(captured_dl):.4f}  Acc: {acc:.2f}%")

print(f"\n[INFO] 最佳准确率: {best_acc:.2f}%")
print(f"[INFO] 模型已保存: ./models/model_finetuned.pkl")
