# -*- coding: utf-8 -*- #

# -----------------------------------------------------------------------
# File Name:    model.py
# Version:      ver1_0
# Created:      2024/06/17
# Description:  本文件定义了CustomNet类，用于定义神经网络模型
#               ★★★请在空白处填写适当的语句，将CustomNet类的定义补充完整★★★
# -----------------------------------------------------------------------

import torch
from torch import nn


class CustomNet(nn.Module):
    """自定义神经网络模型。
    请完成对__init__、forward方法的实现，以完成CustomNet类的定义。

    模型架构说明：
    采用CNN卷积神经网络进行手势图片分类。
    - 输入: 3×64×64 RGB彩色图片
    - 4个卷积块，每块包含 Conv2d → BatchNorm → ReLU → MaxPool
    - 全连接分类器: 2层MLP + Dropout
    - 输出: 6个类别（手势数字0-5）
    """

    def __init__(self, num_classes=6):
        """初始化方法。
        在本方法中，请完成神经网络的各个模块/层的定义。
        请确保每层的输出维度与下一层的输入维度匹配。
        """
        super(CustomNet, self).__init__()

        # START----------------------------------------------------------
        # 卷积特征提取层
        self.features = nn.Sequential(
            # Block 1: 3×64×64 → 32×32×32
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                       # 32×32×32

            # Block 2: 32×32×32 → 64×16×16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                       # 64×16×16

            # Block 3: 64×16×16 → 128×8×8
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                       # 128×8×8

            # Block 4: 128×8×8 → 256×4×4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                       # 256×4×4
        )

        # 全连接分类器
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),             # 256×4×4 → 256×1×1
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )
        # END------------------------------------------------------------

    def forward(self, x):
        """前向传播过程。
        在本方法中，请完成对神经网络前向传播计算的定义。
        """
        # START----------------------------------------------------------
        # 卷积特征提取
        x = self.features(x)

        # 分类器（含全局池化 + 展平 + 全连接）
        x = self.classifier(x)
        return x
        # END------------------------------------------------------------


if __name__ == "__main__":
    # 测试
    from dataset import CustomDataset
    from torchvision.transforms import ToTensor

    c = CustomDataset('./images/train.txt', './images/train', ToTensor)
    net = CustomNet()                                # 实例化
    x = torch.unsqueeze(c[10]['image'], 0)      # 模拟一个模型的输入数据
    print(net.forward(x))                            # 测试forward方法
