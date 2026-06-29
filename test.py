# -*- coding: utf-8 -*- #

# -----------------------------------------------------------------------
# File Name:    test.py
# Version:      ver1_0
# Created:      2024/06/17
# Description:  本文件定义了模型的测试流程
#               ★★★请在空白处填写适当的语句，将模型测试流程补充完整★★★
# -----------------------------------------------------------------------

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from dataset import CustomDataset


def test(dataloader, model, device):
    """定义测试流程。
    :param dataloader: 数据加载器
    :param model: 训练好的模型
    :param device: 测试使用的设备，即使用哪一块CPU、GPU进行模型测试
    """
    # 将模型置为评估（测试）模式
    model.eval()

    size = len(dataloader.dataset)  # 测试集样本总数
    correct_num = 0                 # 预测正确的样本数

    # START----------------------------------------------------------
    # 禁用梯度计算，节省内存和计算资源
    with torch.no_grad():
        for batch_data in dataloader:
            # 从DataLoader返回的字典中获取图像和标签
            X = batch_data['image']
            y = batch_data['label']

            # 将数据移动到测试设备
            X, y = X.to(device), y.to(device)

            # 前向传播
            pred = model(X)

            # 统计预测正确的样本数
            correct_num += (pred.argmax(dim=1) == y).sum().item()

    # 计算并输出测试准确率
    accuracy = correct_num / size * 100
    print(f"\n========== 测试结果 ==========")
    print(f"测试集样本数: {size}")
    print(f"预测正确数:   {correct_num}")
    print(f"测试准确率:   {accuracy:.2f}%")
    print(f"===============================\n")
    # END------------------------------------------------------------


if __name__ == "__main__":
    # 加载训练好的模型
    model = torch.load('./models/model.pkl', weights_only=False)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model.to(device)

    # 测试数据加载器
    test_dataloader = DataLoader(CustomDataset('./images/test.txt', './images/test', ToTensor),
                                 batch_size=32)
    # 运行测试函数
    test(test_dataloader, model, device)
