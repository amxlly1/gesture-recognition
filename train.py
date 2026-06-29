# -*- coding: utf-8 -*- #

# -----------------------------------------------------------------------
# File Name:    train.py
# Version:      ver1_0
# Created:      2024/06/17
# Description:  本文件定义了模型的训练流程
#               ★★★请在空白处填写适当的语句，将模型训练流程补充完整★★★
# -----------------------------------------------------------------------

import torch
from torch import nn
from torchvision.transforms import ToTensor, Compose, RandomHorizontalFlip, RandomRotation, ColorJitter
from torch.utils.data import DataLoader
from dataset import CustomDataset
from model import CustomNet


def train_loop(epoch, dataloader, model, loss_fn, optimizer, scheduler, device):
    """定义训练流程。
    :param epoch: 定义训练的总轮次
    :param dataloader: 数据加载器
    :param model: 模型，需在model.py文件中定义好
    :param loss_fn: 损失函数
    :param optimizer: 优化器
    :param scheduler: 学习率调度器
    :param device: 训练设备，即使用哪一块CPU、GPU进行训练
    """
    # START----------------------------------------------------------
    # 获取数据集总大小
    total_samples = len(dataloader.dataset)

    for i in range(epoch):
        model.train()
        total_loss = 0.0
        correct = 0

        for batch_data in dataloader:
            # 从DataLoader返回的字典中获取图像和标签
            X = batch_data['image']
            y = batch_data['label']

            # 将数据移动到训练设备
            X, y = X.to(device), y.to(device)

            # 前向传播：计算预测值
            pred = model(X)

            # 计算损失
            loss = loss_fn(pred, y)

            # 反向传播：清空梯度 → 计算梯度 → 更新参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 累计损失和正确数
            total_loss += loss.item()
            correct += (pred.argmax(dim=1) == y).sum().item()

        # 更新学习率
        scheduler.step()

        # 每轮输出训练损失和准确率
        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total_samples * 100
        print(f"Epoch [{i + 1:>3d}/{epoch}]  "
              f"Loss: {avg_loss:.4f}  "
              f"Accuracy: {accuracy:.2f}%  "
              f"({correct}/{total_samples})")
    # END------------------------------------------------------------

    # 保存模型
    torch.save(model, './models/model.pkl')


if __name__ == "__main__":
    # 定义模型超参数
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    EPOCH = 50

    # 模型实例化
    model = CustomNet()
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model.to(device)

    # 训练数据加载器（使用数据增强 + shuffle）
    # 注意：CustomDataset的transform参数需要是可调用对象（类），调用transform()完成实例化
    # 因此用lambda包装Compose
    train_dataloader = DataLoader(
        CustomDataset('./images/train.txt', './images/train',
                      lambda: Compose([
                          RandomHorizontalFlip(p=0.5),
                          RandomRotation(degrees=15),
                          ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                          ToTensor(),
                      ])),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    # 损失函数
    loss_fn = nn.CrossEntropyLoss()
    # Adam 优化器（自适应学习率，比纯SGD更好）
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # 余弦退火学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCH)

    # 调用训练方法
    train_loop(EPOCH, train_dataloader, model, loss_fn, optimizer, scheduler, device)
