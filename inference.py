# -*- coding: utf-8 -*- #

# -----------------------------------------------------------------------
# File Name:    inference.py
# Version:      ver1_0
# Created:      2024/06/17
# Description:  本文件定义了用于在模型应用端进行推理，返回模型输出的流程
#               ★★★请在空白处填写适当的语句，将模型推理应用流程补充完整★★★
# -----------------------------------------------------------------------

import torch
from PIL import Image
from torchvision.transforms import ToTensor


def inference(image_path, model, device):
    """定义模型推理应用的流程。
    :param image_path: 输入图片的路径
    :param model: 训练好的模型
    :param device: 模型推理使用的设备，即使用哪一块CPU、GPU进行模型推理
    """
    # 将模型置为评估（测试）模式
    model.eval()

    # START----------------------------------------------------------
    # 加载图片并转换为Tensor
    image = Image.open(image_path).convert('RGB')

    # 应用与训练时相同的预处理（ToTensor会将像素值归一化到[0,1]）
    transform = ToTensor()
    image_tensor = transform(image)

    # 增加batch维度：(C, H, W) → (1, C, H, W)
    image_tensor = image_tensor.unsqueeze(0).to(device)

    # 禁用梯度进行推理
    with torch.no_grad():
        output = model(image_tensor)

        # 获取预测类别（输出logits中最大值对应的索引）
        predicted_class = output.argmax(dim=1).item()

        # 获取各类别的概率（softmax）
        probabilities = torch.softmax(output, dim=1).squeeze(0)

    # 输出预测结果
    print(f"\n========== 推理结果 ==========")
    print(f"输入图片: {image_path}")
    print(f"预测数字: {predicted_class}")
    print(f"\n各类别概率:")
    for i, prob in enumerate(probabilities):
        bar = "█" * int(prob.item() * 40)
        print(f"  数字 {i}: {prob.item():.4f}  {bar}")
    print(f"===============================\n")
    # END------------------------------------------------------------


if __name__ == "__main__":
    # 指定图片路径
    image_path = "images/test/signs/img_0115.png"

    # 加载训练好的模型
    model = torch.load('./models/model.pkl', weights_only=False)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model.to(device)

    # 显示图片，输出预测结果
    inference(image_path, model, device)
