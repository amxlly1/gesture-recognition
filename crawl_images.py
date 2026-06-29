# -*- coding: utf-8 -*-
"""
从 Bing 图片搜索爬取手势数字图片 (0-5)
每类下载 150 张，保存到 crawled/ 目录
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from icrawler.builtin import BingImageCrawler
from PIL import Image

SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'crawled')
os.makedirs(SAVE_DIR, exist_ok=True)

# 每类搜索关键词（中英文混合增加多样性）
queries = {
    0: [
        'hand gesture zero fingers closed fist',
        '手势 数字0 握拳',
        'hand showing zero count fingers',
    ],
    1: [
        'hand gesture number one index finger up',
        '手势 数字1 食指',
        'hand showing one count finger',
    ],
    2: [
        'hand gesture number two fingers peace',
        '手势 数字2 两根手指',
        'hand showing two count fingers',
    ],
    3: [
        'hand gesture number three fingers',
        '手势 数字3 三根手指',
        'hand showing three count fingers',
    ],
    4: [
        'hand gesture number four fingers',
        '手势 数字4 四根手指',
        'hand showing four count fingers',
    ],
    5: [
        'hand gesture number five fingers open palm',
        '手势 数字5 五指张开',
        'hand showing five count palm',
    ],
}

MAX_PER_CLASS = 150

for class_id in range(6):
    class_dir = os.path.join(SAVE_DIR, str(class_id))
    os.makedirs(class_dir, exist_ok=True)

    for keyword in queries[class_id]:
        # 检查是否已够数
        existing = len(os.listdir(class_dir))
        if existing >= MAX_PER_CLASS:
            break

        crawler = BingImageCrawler(
            downloader_threads=4,
            storage={'root_dir': class_dir}
        )

        # 下载（icrawler 会自动加 000001 后缀，不会覆盖）
        crawler.crawl(
            keyword=keyword,
            max_num=MAX_PER_CLASS,
            min_size=(50, 50),    # 只下载 >= 50x50 的
            file_idx_offset=existing
        )

        print(f"  [Class {class_id}] '{keyword}' → {len(os.listdir(class_dir))} 张")

print("\n--- 下载完成 ---")
for i in range(6):
    count = len(os.listdir(os.path.join(SAVE_DIR, str(i))))
    print(f"  手势 {i}: {count} 张")

# 清理损坏图片
print("\n--- 清理损坏图片 ---")
bad = 0
for i in range(6):
    class_dir = os.path.join(SAVE_DIR, str(i))
    for fname in os.listdir(class_dir):
        fpath = os.path.join(class_dir, fname)
        try:
            img = Image.open(fpath)
            img.verify()
        except:
            os.remove(fpath)
            bad += 1
            print(f"  删除损坏: {fpath}")
print(f"  共删除 {bad} 张损坏图片")

# 生成 labels.txt
print("\n--- 生成标签文件 ---")
label_lines = []
for i in range(6):
    class_dir = os.path.join(SAVE_DIR, str(i))
    for fname in os.listdir(class_dir):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp')):
            rel_path = f"crawled/{i}/{fname}"
            label_lines.append(f"{rel_path} {i}")

labels_path = os.path.join(SAVE_DIR, 'labels.txt')
with open(labels_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(label_lines))

print(f"  标签文件: {labels_path}")
print(f"  总计: {len(label_lines)} 张")
