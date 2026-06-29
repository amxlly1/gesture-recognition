# -*- coding: utf-8 -*-
"""从 Kaggle 下载手势数据集并整理成训练格式"""
import kagglehub
import os, shutil
from PIL import Image

BASE = r"D:\py\神经网络实训\nndl_project"
EXTERNAL = os.path.join(BASE, 'external_data')
os.makedirs(EXTERNAL, exist_ok=True)

# ====== 数据集1: koryakinp/fingers (约500张/类，纯手部) ======
print("=== 下载 koryakinp/fingers ===")
try:
    path1 = kagglehub.dataset_download("koryakinp/fingers")
    print(f"  下载到: {path1}")
    # 复制到 external_data
    dst = os.path.join(EXTERNAL, 'fingers_kaggle')
    if os.path.exists(dst): shutil.rmtree(dst)
    shutil.copytree(path1, dst)
    print(f"  已复制到: {dst}")
except Exception as e:
    print(f"  下载失败: {e}")

# ====== 数据集2: rosÃ©a6/finger-digits-05 (12000张!) ======
print("\n=== 下载 rosÃ©a6/finger-digits-05 ===")
try:
    path2 = kagglehub.dataset_download("roshea6/finger-digits-05")
    print(f"  下载到: {path2}")
    dst = os.path.join(EXTERNAL, 'finger_digits_05')
    if os.path.exists(dst): shutil.rmtree(dst)
    shutil.copytree(path2, dst)
    print(f"  已复制到: {dst}")
except Exception as e:
    print(f"  下载失败: {e}")

print("\n=== 下载完成 ===")
print(f"数据目录: {EXTERNAL}")
for d in os.listdir(EXTERNAL):
    full = os.path.join(EXTERNAL, d)
    count = sum(1 for _ in os.listdir(full)) if os.path.isdir(full) else 0
    print(f"  {d}: {count} 个文件")
