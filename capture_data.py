# -*- coding: utf-8 -*-
"""
手势数据采集工具
在摄像头前做出手势，按数字键 0-5 采集，按 q 退出。
采集的图片保存到 captured/ 目录，标签写入 captured_labels.txt
"""

import cv2
import os
import sys
from PIL import Image
import numpy as np

# 采集保存路径（使用绝对路径避免工作目录问题）
SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'captured')
os.makedirs(SAVE_DIR, exist_ok=True)
print(f"[INFO] 保存目录: {SAVE_DIR}")

LABELS_FILE = os.path.join(SAVE_DIR, 'labels.txt')

print("=" * 60)
print("  手势数据采集工具")
print("=" * 60)
print()
print("  操作说明：")
print("    将手放入画面中央的绿色框内")
print("    按数字键 0-5 采集对应手势")
print("    按 q 退出")
print(f"    图片保存在: {SAVE_DIR}")
print()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] 无法打开摄像头！")
    sys.exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# 采集计数器
counts = {str(i): 0 for i in range(6)}

# 打开标签文件（追加模式）
label_f = open(LABELS_FILE, 'a', encoding='utf-8')

print("开始采集...\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # 镜像翻转
    h, w = frame.shape[:2]

    # 绘制中央 ROI 引导框
    roi_size = min(w, h) // 2
    x1 = (w - roi_size) // 2
    y1 = (h - roi_size) // 2
    x2 = x1 + roi_size
    y2 = y1 + roi_size
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 显示采集计数
    y_pos = 30
    for i in range(6):
        text = f"Gesture {i}: {counts[str(i)]}"
        cv2.putText(frame, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0) if i < 6 else (0, 0, 255), 2)
        y_pos += 25

    cv2.putText(frame, "Press 0-5 to capture | q to quit",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow('Gesture Data Capture - Press 0-5 to capture, q to quit', frame)

    key = cv2.waitKey(30) & 0xFF

    if key == ord('q'):
        break
    elif key >= ord('0') and key <= ord('5'):
        gesture_num = chr(key)

        # 裁剪中心 ROI
        cropped = frame[y1:y2, x1:x2]

        # 保存图片（用 PIL 绕过 OpenCV 中文路径 bug）
        filename = f"gesture_{gesture_num}_{counts[gesture_num]:04d}.png"
        filepath = os.path.join(SAVE_DIR, filename)
        cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(cropped_rgb)
        pil_img.save(filepath, 'PNG')

        # 写入标签
        label_f.write(f"./captured/{filename} {gesture_num}\n")
        label_f.flush()

        counts[gesture_num] += 1
        print(f"  采集: {filename}  (手势 {gesture_num} 共 {counts[gesture_num]} 张)")

        # 采集后短暂闪烁提示
        flash = frame.copy()
        flash[y1:y2, x1:x2] = (0, 255, 0)
        cv2.addWeighted(flash, 0.3, frame, 0.7, 0, frame)
        cv2.imshow('Gesture Data Capture - Press 0-5 to capture, q to quit', frame)
        cv2.waitKey(200)

cap.release()
label_f.close()
cv2.destroyAllWindows()

print("\n" + "=" * 60)
print("  采集完成！")
print("=" * 60)
for i in range(6):
    print(f"  手势 {i}: {counts[str(i)]} 张")
total = sum(counts.values())
print(f"  总计: {total} 张")
print(f"\n  数据保存在: {SAVE_DIR}")
print(f"  标签文件: {LABELS_FILE}")
print(f"\n  下一步运行: python finetune.py 进行微调训练")
