# -*- coding: utf-8 -*-
"""
手势数字识别 - 实时交互应用
Flask Web 后端：加载训练好的模型，提供图片上传识别 API
"""

import io
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from flask import Flask, request, jsonify, render_template
from torchvision.transforms import ToTensor
import sys
import os

# 确保能导入同目录下的 model.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import CustomNet

app = Flask(__name__)

# ---------- 加载模型（启动时执行一次） ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] 使用设备: {device}")

model_path = os.path.join(os.path.dirname(__file__), 'models', 'model.pkl')
model = torch.load(model_path, map_location=device, weights_only=False)
model.to(device)
model.eval()
print("[INFO] 模型加载完成，就绪。")

# 与训练时保持一致的预处理
transform = ToTensor()

# ---------- 置信度阈值（低于此值认为无手势） ----------
CONFIDENCE_THRESHOLD = 0.70     # 单一类别最大概率低于70% → 无手势
ENTROPY_THRESHOLD = 1.20        # 熵高于1.2（分布太均匀） → 无手势


def detect_skin_ratio(img_np):
    """检测图片中肤色像素的比例（基于YCbCr色彩空间）。

    肤色在YCbCr空间中有较好的聚类特性：
    77 <= Cb <= 127, 133 <= Cr <= 173

    :param img_np: numpy array (64, 64, 3) RGB
    :return: 肤色像素占比 (0.0 ~ 1.0)
    """
    # RGB → YCbCr
    r, g, b = img_np[:, :, 0].astype(np.float32), img_np[:, :, 1].astype(np.float32), img_np[:, :, 2].astype(np.float32)
    Y = 0.299 * r + 0.587 * g + 0.114 * b
    Cb = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b
    Cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b

    # 肤色范围
    skin_mask = (Cb >= 77) & (Cb <= 127) & (Cr >= 133) & (Cr <= 173)
    return skin_mask.mean()


def compute_entropy(probs):
    """计算概率分布的熵。

    熵越高说明模型越不确定——各分类概率越平均。
    熵越低说明模型越确信——某个分类概率极高。

    :param probs: torch.Tensor, shape (6,)
    :return: float, 熵值
    """
    # 避免 log(0)，加极小值
    eps = 1e-10
    log_probs = torch.log(probs + eps)
    entropy = -torch.sum(probs * log_probs).item()
    return entropy


@app.route('/')
def index():
    """主页面：摄像头实时识别 + 上传图片识别"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """接收上传的图片，返回预测结果"""
    if 'image' not in request.files:
        return jsonify({'error': '未收到图片'}), 400

    file = request.files['image']

    try:
        # 读取图片并转换为 RGB
        img = Image.open(io.BytesIO(file.read())).convert('RGB')

        # 调整尺寸为 64x64（与训练数据一致）
        img = img.resize((64, 64), Image.BILINEAR)

        # ---- 肤色检测（在转 Tensor 之前用 numpy 做） ----
        img_np = np.array(img, dtype=np.float32) / 255.0
        skin_ratio = detect_skin_ratio(img_np)

        # 预处理：ToTensor → 归一化到 [0,1] → 增加 batch 维度
        img_tensor = transform(img).unsqueeze(0).to(device)

        # 推理
        with torch.no_grad():
            output = model(img_tensor)
            probs = F.softmax(output, dim=1).squeeze(0)
            pred_class = output.argmax(dim=1).item()
            max_conf = probs[pred_class].item()

        # 计算熵值
        entropy = compute_entropy(probs)

        # ---- 综合判断是否为有效手势 ----
        # 两个条件同时满足才认为检测到手势：
        #   1. 最大概率 >= 阈值（模型足够确信）
        #   2. 熵 <= 阈值（概率分布集中而非均匀）
        is_gesture = bool(
            max_conf >= CONFIDENCE_THRESHOLD
            and entropy <= ENTROPY_THRESHOLD
        )

        # 构建返回结果
        probabilities = {str(i): round(probs[i].item(), 4) for i in range(6)}

        return jsonify({
            'prediction': pred_class if is_gesture else None,
            'confidence': round(max_conf * 100, 2),
            'probabilities': probabilities,
            'is_gesture': is_gesture,
            'entropy': round(entropy, 4),
            'skin_ratio': round(skin_ratio, 4),
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False)
