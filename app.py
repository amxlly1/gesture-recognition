# -*- coding: utf-8 -*-
"""
手势数字识别 - 实时交互应用 (v3)
Flask Web 后端 + 多策略手部检测 + CNN 分类

v3 改进：
- 多色彩空间融合肤色检测（YCrCb + HSV 双通道互补）
- 更宽松的肤色阈值（覆盖更多肤色类型和光照条件）
- 降级策略：肤色检测失败时使用全图 + 更严格的置信度阈值
"""

import io
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from flask import Flask, request, jsonify, render_template
from torchvision.transforms import ToTensor
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import CustomNet

app = Flask(__name__)

# ---------- 加载模型 ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] 使用设备: {device}")

model_path = os.path.join(os.path.dirname(__file__), 'models', 'model_final.pkl')
model = torch.load(model_path, map_location=device, weights_only=False)
model.to(device)
model.eval()
print("[INFO] 模型加载完成，就绪。")

transform = ToTensor()

# ---------- 阈值 ----------
CONFIDENCE_THRESHOLD = 0.50
ENTROPY_THRESHOLD = 1.50
# 无手部检测时的更高阈值
CONFIDENCE_THRESHOLD_NO_HAND = 0.95


def detect_hand_roi_v2(img_bgr):
    """v2 多策略手部检测。

    融合 YCrCb 和 HSV 两个色彩空间的肤色检测结果，
    加上更宽松的阈值 + 自适应光照补偿。

    :param img_bgr: BGR numpy array
    :return: (cropped_img, bbox) 或 (None, None)
    """
    h, w = img_bgr.shape[:2]

    # ---- 策略1: YCrCb 肤色（更宽范围） ----
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    lower1 = np.array([0, 130, 70], dtype=np.uint8)    # Cr放宽到130, Cb放宽到70
    upper1 = np.array([255, 180, 135], dtype=np.uint8)  # Cr到180, Cb到135
    mask1 = cv2.inRange(ycrcb, lower1, upper1)

    # ---- 策略2: HSV 肤色（辅助） ----
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower2 = np.array([0, 15, 50], dtype=np.uint8)     # 低饱和度
    upper2 = np.array([30, 180, 255], dtype=np.uint8)   # 涵盖暖色+暗肤
    mask2 = cv2.inRange(hsv, lower2, upper2)

    # ---- 融合两个mask（任一检测到就算） ----
    skin_mask = cv2.bitwise_or(mask1, mask2)

    # ---- 形态学 ----
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # ---- 找轮廓 ----
    contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    # 取前3大轮廓中面积最大的（可能手被分割成多块）
    contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)
    best = contours_sorted[0]

    # 合并前3个靠近的轮廓（手可能因肤色不均被断开）
    if len(contours_sorted) >= 2:
        bx, by, bw_, bh_ = cv2.boundingRect(best)
        for cnt in contours_sorted[1:min(4, len(contours_sorted))]:
            nx, ny, nw, nh = cv2.boundingRect(cnt)
            # 如果两个框重叠或靠近，合并
            if (abs((bx+bw_/2) - (nx+nw/2)) < max(bw_, nw)*2 and
                abs((by+bh_/2) - (ny+nh/2)) < max(bh_, nh)*2):
                bx = min(bx, nx); by = min(by, ny)
                bw_ = max(bx+bw_, nx+nw) - bx
                bh_ = max(by+bh_, ny+nh) - by
                best = np.vstack([best, cnt])

    x, y, bw, bh = cv2.boundingRect(best)
    area = bw * bh
    min_area = w * h * 0.008  # 降到0.8%（更敏感）
    if area < min_area:
        return None, None

    # 扩展为正方形，扩大10%边距
    cx, cy = x + bw // 2, y + bh // 2
    side = int(max(bw, bh) * 1.3)
    side = min(side, max(w, h))

    x1 = max(0, cx - side // 2)
    y1 = max(0, cy - side // 2)
    x2 = min(w, x1 + side)
    y2 = min(h, y1 + side)

    cropped = img_bgr[y1:y2, x1:x2]
    return cropped, (x1, y1, x2, y2)


def compute_entropy(probs):
    eps = 1e-10
    return -torch.sum(probs * torch.log(probs + eps)).item()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': '未收到图片'}), 400

    file = request.files['image']

    try:
        img_bytes = file.read()
        pil_img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        img_np = np.array(pil_img)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # ---- 手部检测 ----
        hand_crop, bbox = detect_hand_roi_v2(img_bgr)

        if hand_crop is not None:
            # 检测到手 → 裁剪送入
            hand_rgb = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2RGB)
        else:
            # 没检测到手 → 用全图，但提高置信度阈值
            hand_rgb = img_np

        hand_pil = Image.fromarray(hand_rgb).resize((64, 64), Image.BILINEAR)
        img_tensor = transform(hand_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_tensor)
            probs = F.softmax(output, dim=1).squeeze(0)
            pred_class = output.argmax(dim=1).item()
            max_conf = probs[pred_class].item()

        entropy = compute_entropy(probs)

        # 动态阈值：有手检测用88%，无手检测用95%
        threshold = CONFIDENCE_THRESHOLD if bbox else CONFIDENCE_THRESHOLD_NO_HAND
        is_gesture = bool(max_conf >= threshold and entropy <= ENTROPY_THRESHOLD)

        probabilities = {str(i): round(probs[i].item(), 4) for i in range(6)}

        return jsonify({
            'prediction': pred_class if is_gesture else None,
            'confidence': round(max_conf * 100, 2),
            'probabilities': probabilities,
            'is_gesture': is_gesture,
            'entropy': round(entropy, 4),
            'hand_detected': bbox is not None,
            'bbox': list(bbox) if bbox else None,
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False)
