# -*- coding: utf-8 -*-
"""
手势数字识别 - 实时交互应用
Flask Web 后端：加载训练好的模型，提供图片上传识别 API
"""

import io
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

LABELS = ['手势 0', '手势 1', '手势 2', '手势 3', '手势 4', '手势 5']


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

        # 预处理：ToTensor → 归一化到 [0,1] → 增加 batch 维度
        img_tensor = transform(img).unsqueeze(0).to(device)

        # 推理
        with torch.no_grad():
            output = model(img_tensor)
            probs = F.softmax(output, dim=1).squeeze(0)
            pred_class = output.argmax(dim=1).item()

        # 构建返回结果
        probabilities = {str(i): round(probs[i].item(), 4) for i in range(6)}

        return jsonify({
            'prediction': pred_class,
            'confidence': round(probs[pred_class].item() * 100, 2),
            'probabilities': probabilities
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False)
