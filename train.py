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
from torchvision.transforms import ToTensor, Compose, RandomHorizontalFlip, RandomRotation, ColorJitter, RandomAffine, RandomPerspective
from torch.utils.data import DataLoader
from dataset import CustomDataset
from model import CustomNet
import random
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np


class RandomBackgroundReplace:
    """随机背景替换增强。
    将手势图片的浅色背景替换为随机纯色/渐变/噪声背景，
    模拟真实摄像头拍摄环境，提升模型对背景的鲁棒性。
    """

    def __init__(self, p=0.6):
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img

        # 生成随机背景
        bg_type = random.choice(['solid', 'gradient', 'noise', 'keep'])
        w, h = img.size

        if bg_type == 'solid':
            bg = Image.new('RGB', (w, h), (
                random.randint(30, 225),
                random.randint(30, 225),
                random.randint(30, 225)
            ))
        elif bg_type == 'gradient':
            c1 = (random.randint(30, 225), random.randint(30, 225), random.randint(30, 225))
            c2 = (random.randint(30, 225), random.randint(30, 225), random.randint(30, 225))
            direction = random.choice(['horizontal', 'vertical', 'diagonal'])
            bg = Image.new('RGB', (w, h))
            for y in range(h):
                for x in range(w):
                    if direction == 'horizontal':
                        ratio = x / w
                    elif direction == 'vertical':
                        ratio = y / h
                    else:
                        ratio = (x + y) / (w + h)
                    r = int(c1[0] + (c2[0] - c1[0]) * ratio)
                    g = int(c1[1] + (c2[1] - c1[1]) * ratio)
                    b = int(c1[2] + (c2[2] - c1[2]) * ratio)
                    bg.putpixel((x, y), (r, g, b))
        elif bg_type == 'noise':
            arr = np.random.randint(30, 225, (h, w, 3), dtype=np.uint8)
            bg = Image.fromarray(arr)
            bg = bg.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 2.0)))
        else:  # keep
            return img

        # 手势区域检测：基于亮度阈值找到手势前景（非背景部分）
        arr = np.array(img.convert('RGB'), dtype=np.float32)
        # 计算每个像素的亮度，找出手势区域（手部通常比白背景暗）
        brightness = arr.mean(axis=2)
        # 亮度大于阈值的视为背景，替换为随机背景
        threshold = 180.0 + random.uniform(-20, 20)
        bg_arr = np.array(bg, dtype=np.float32)
        mask = brightness < threshold  # True = 前景（手）, False = 背景
        mask_3d = np.stack([mask] * 3, axis=2)
        # 用随机背景替换原背景
        result_arr = np.where(mask_3d, arr, bg_arr)
        result = Image.fromarray(result_arr.astype(np.uint8), 'RGB')

        # 随机调整亮度/对比度进一步增加多样性
        if random.random() < 0.3:
            result = ImageEnhance.Brightness(result).enhance(random.uniform(0.7, 1.3))
        if random.random() < 0.3:
            result = ImageEnhance.Contrast(result).enhance(random.uniform(0.7, 1.3))

        return result


class RandomBlur:
    """随机模糊，模拟摄像头失焦"""
    def __init__(self, p=0.15):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
        return img


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
    total_samples = len(dataloader.dataset)

    for i in range(epoch):
        model.train()
        total_loss = 0.0
        correct = 0

        for batch_data in dataloader:
            X = batch_data['image']
            y = batch_data['label']
            X, y = X.to(device), y.to(device)

            pred = model(X)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (pred.argmax(dim=1) == y).sum().item()

        scheduler.step()

        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total_samples * 100
        print(f"Epoch [{i + 1:>3d}/{epoch}]  "
              f"Loss: {avg_loss:.4f}  "
              f"Accuracy: {accuracy:.2f}%  "
              f"({correct}/{total_samples})")
    # END------------------------------------------------------------

    torch.save(model, './models/model.pkl')


if __name__ == "__main__":
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    EPOCH = 50

    model = CustomNet()
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model.to(device)

    # 强力数据增强：背景替换 + 几何变换 + 颜色抖动 + 模糊
    train_dataloader = DataLoader(
        CustomDataset('./images/train.txt', './images/train',
                      lambda: Compose([
                          RandomBackgroundReplace(p=0.65),     # 65% 概率替换背景
                          RandomRotation(degrees=25),           # 旋转 ±25°
                          RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.85, 1.15)),  # 平移+缩放
                          RandomHorizontalFlip(p=0.5),
                          ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),  # 颜色抖动
                          RandomBlur(p=0.15),                   # 随机模糊
                          RandomPerspective(distortion_scale=0.1, p=0.2),  # 透视变换
                          ToTensor(),
                      ])),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCH)

    train_loop(EPOCH, train_dataloader, model, loss_fn, optimizer, scheduler, device)
