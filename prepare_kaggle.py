# -*- coding: utf-8 -*-
"""解析 Kaggle fingers 数据集，生成统一格式的 labels.txt"""
import os, re, shutil

KAGGLE_DIR = r"C:\Users\刘宇\.cache\kagglehub\datasets\koryakinp\fingers\versions\2"
DEST_DIR = r"D:\py\神经网络实训\nndl_project\external_data"
os.makedirs(DEST_DIR, exist_ok=True)

# 复制图片到 external_data 并用标签分类（方便排查）
# 同时生成 labels.txt
all_labels = []
total = 0

for split in ['train', 'test']:
    split_dir = os.path.join(KAGGLE_DIR, split)
    if not os.path.exists(split_dir):
        continue
    for fname in os.listdir(split_dir):
        # 从文件名提取标签: xxx_0L.png → 0, xxx_5R.png → 5
        match = re.search(r'_(\d)[LR]\.(png|jpg|jpeg)', fname, re.IGNORECASE)
        if not match:
            continue
        label = match.group(1)

        src = os.path.join(split_dir, fname)
        dst_dir = os.path.join(DEST_DIR, label)
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, fname)

        # 如果已存在跳过
        if not os.path.exists(dst):
            shutil.copy2(src, dst)

        # 相对路径标签
        rel = f"external_data/{label}/{fname}"
        all_labels.append(f"{rel} {label}")
        total += 1

# 写入 labels.txt
labels_path = os.path.join(DEST_DIR, 'labels.txt')
with open(labels_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(all_labels))

print(f"总计: {total} 张")
for i in range(6):
    cnt = len(os.listdir(os.path.join(DEST_DIR, str(i))))
    print(f"  手势 {i}: {cnt} 张")
print(f"\n标签文件: {labels_path}")
