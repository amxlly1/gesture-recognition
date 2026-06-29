"""用 Pillow 生成报告插图（不依赖 matplotlib）"""
from PIL import Image, ImageDraw, ImageFont
import os
import numpy as np

SAVE_DIR = r'D:\py\神经网络实训\nndl_project\report_images'
os.makedirs(SAVE_DIR, exist_ok=True)

# 尝试加载中文字体
font_paths = [
    'C:/Windows/Fonts/msyh.ttc',
    'C:/Windows/Fonts/simhei.ttf',
    'C:/Windows/Fonts/simsun.ttc',
]
font_title = None
font_normal = None
for fp in font_paths:
    if os.path.exists(fp):
        try:
            font_title = ImageFont.truetype(fp, 28)
            font_normal = ImageFont.truetype(fp, 20)
            font_small = ImageFont.truetype(fp, 15)
            font_tiny = ImageFont.truetype(fp, 13)
            break
        except:
            pass

if font_title is None:
    font_title = ImageFont.load_default()
    font_normal = ImageFont.load_default()
    font_small = ImageFont.load_default()
    font_tiny = ImageFont.load_default()

def create_confusion_matrix():
    """混淆矩阵热力图"""
    confusion = np.array([
        [80,  0,  0,  0,  0,  0],
        [ 0, 80,  0,  0,  0,  0],
        [ 0,  4, 76,  0,  0,  0],
        [ 0,  0, 12, 68,  0,  0],
        [ 0,  0,  0,  0, 80,  0],
        [ 0,  0,  0,  0,  0, 80],
    ])

    cell_size = 70
    margin_left = 80
    margin_top = 100
    margin_right = 50
    margin_bottom = 50

    w = margin_left + 6 * cell_size + margin_right
    h = margin_top + 6 * cell_size + margin_bottom + 40
    img = Image.new('RGB', (w, h), 'white')
    draw = ImageDraw.Draw(img)

    # 标题
    draw.text((w//2 - 250, 20), '手势识别混淆矩阵', fill='#1a1a2e', font=font_title)
    draw.text((w//2 - 200, 58), '(测试集 480 张, 总准确率 97.08%)', fill='#666', font=font_small)

    max_val = confusion.max()

    for i in range(6):
        for j in range(6):
            x = margin_left + j * cell_size
            y = margin_top + i * cell_size
            val = confusion[i, j]
            # 颜色越深代表值越大
            ratio = val / max_val if max_val > 0 else 0
            r = int(235 - 150 * ratio)
            g = int(240 - 140 * ratio)
            b = int(250 - 120 * ratio)
            draw.rectangle([x, y, x + cell_size, y + cell_size], fill=(r, g, b), outline='#ccc', width=1)
            # 数值
            text_color = 'white' if ratio > 0.5 else '#333'
            tw = font_normal.getbbox(str(val))[2] if hasattr(font_normal, 'getbbox') else font_normal.getsize(str(val))[0]
            draw.text((x + cell_size//2 - tw//2, y + cell_size//2 - 14), str(val), fill=text_color, font=font_normal)

    # 行列标签
    for i in range(6):
        y = margin_top + i * cell_size + cell_size // 2 - 10
        draw.text((10, y), f'真实{i}', fill='#333', font=font_normal)
        x = margin_left + i * cell_size + cell_size // 2 - 25
        draw.text((x, margin_top + 6 * cell_size + 8), f'预测{i}', fill='#333', font=font_small)

    # 标注错误对
    draw.text((margin_left + 6*cell_size + 10, margin_top + 2*cell_size + 10), '2→1: 4张', fill='#e74c3c', font=font_tiny)
    draw.text((margin_left + 6*cell_size + 10, margin_top + 3*cell_size + 10), '3→2: 12张', fill='#e74c3c', font=font_tiny)

    img.save(os.path.join(SAVE_DIR, 'confusion_matrix.png'))
    print('图1: confusion_matrix.png 完成')

def create_training_curve():
    """训练曲线"""
    w, h = 900, 500
    img = Image.new('RGB', (w, h), 'white')
    draw = ImageDraw.Draw(img)

    draw.text((w//2 - 200, 15), '训练过程曲线', fill='#1a1a2e', font=font_title)

    # 数据
    train_acc = [80.42, 88.98, 93.05, 95.67, 96.51, 97.35, 97.72, 97.69, 98.30,
                 98.42, 98.60, 98.75, 98.83, 99.11, 99.08, 99.16, 99.21, 99.44,
                 99.39, 99.55, 99.58, 99.61, 99.68, 99.74, 99.78, 99.78, 99.77,
                 99.84, 99.85, 99.81]
    train_loss = [0.4795, 0.2679, 0.1832, 0.1191, 0.0968, 0.0743, 0.0644, 0.0644,
                  0.0498, 0.0440, 0.0403, 0.0351, 0.0348, 0.0269, 0.0264, 0.0246,
                  0.0224, 0.0167, 0.0171, 0.0135, 0.0125, 0.0123, 0.0099, 0.0084,
                  0.0077, 0.0071, 0.0069, 0.0053, 0.0049, 0.0066]

    # 绘图区域
    plot_left = 80
    plot_right = 420
    plot_top = 60
    plot_bottom = 440

    # Loss 曲线（左侧）
    draw.rectangle([plot_left, plot_top, plot_right, plot_bottom], outline='#ddd')
    draw.text((plot_left + 120, plot_top - 25), '训练损失 (Loss)', fill='#2c3e50', font=font_small)

    max_loss = 0.5
    for i in range(len(train_loss) - 1):
        x1 = plot_left + i * (plot_right - plot_left) // 29
        y1 = plot_bottom - int(train_loss[i] / max_loss * (plot_bottom - plot_top))
        x2 = plot_left + (i+1) * (plot_right - plot_left) // 29
        y2 = plot_bottom - int(train_loss[i+1] / max_loss * (plot_bottom - plot_top))
        draw.line([x1, y1, x2, y2], fill='#3498db', width=2)

    # Y轴标签
    for val in [0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        y = plot_bottom - int(val / max_loss * (plot_bottom - plot_top))
        draw.text((plot_left - 45, y - 8), f'{val:.1f}', fill='#888', font=font_tiny)
    draw.text((plot_left + 120, plot_bottom + 5), 'Epoch', fill='#888', font=font_tiny)
    for e in [1, 10, 20, 30]:
        x = plot_left + (e-1) * (plot_right - plot_left) // 29
        draw.text((x - 5, plot_bottom + 5), str(e), fill='#888', font=font_tiny)

    # Accuracy 曲线（右侧）
    plot_left2 = 520
    plot_right2 = 860
    draw.rectangle([plot_left2, plot_top, plot_right2, plot_bottom], outline='#ddd')
    draw.text((plot_left2 + 100, plot_top - 25), '训练准确率 (%)', fill='#2c3e50', font=font_small)

    for i in range(len(train_acc) - 1):
        x1 = plot_left2 + i * (plot_right2 - plot_left2) // 29
        y1 = plot_bottom - int((train_acc[i] - 75) / 25 * (plot_bottom - plot_top))
        x2 = plot_left2 + (i+1) * (plot_right2 - plot_left2) // 29
        y2 = plot_bottom - int((train_acc[i+1] - 75) / 25 * (plot_bottom - plot_top))
        draw.line([x1, y1, x2, y2], fill='#27ae60', width=2)

    # 测试集准确率参考线
    y_test = plot_bottom - int((97.08 - 75) / 25 * (plot_bottom - plot_top))
    draw.line([plot_left2, y_test, plot_right2, y_test], fill='#e67e22', width=1)
    draw.text((plot_right2 - 130, y_test - 18), '测试集 97.08%', fill='#e67e22', font=font_tiny)

    # Y轴标签
    for val in [80, 85, 90, 95, 100]:
        y = plot_bottom - int((val - 75) / 25 * (plot_bottom - plot_top))
        draw.text((plot_left2 - 35, y - 8), f'{val}%', fill='#888', font=font_tiny)
    draw.text((plot_left2 + 120, plot_bottom + 5), 'Epoch', fill='#888', font=font_tiny)
    for e in [1, 10, 20, 30]:
        x = plot_left2 + (e-1) * (plot_right2 - plot_left2) // 29
        draw.text((x - 5, plot_bottom + 5), str(e), fill='#888', font=font_tiny)

    img.save(os.path.join(SAVE_DIR, 'training_curve.png'))
    print('图2: training_curve.png 完成')

def create_class_bar():
    """各类别准确率柱状图"""
    w, h = 700, 400
    img = Image.new('RGB', (w, h), 'white')
    draw = ImageDraw.Draw(img)

    draw.text((w//2 - 150, 15), '各类别测试准确率', fill='#1a1a2e', font=font_title)

    class_acc = [100, 100, 95.0, 85.0, 100, 100]
    colors = ['#27ae60', '#27ae60', '#f39c12', '#e74c3c', '#27ae60', '#27ae60']
    bar_w = 60
    gap = 30
    start_x = 80
    bottom_y = 320

    for i, (acc, color) in enumerate(zip(class_acc, colors)):
        x = start_x + i * (bar_w + gap)
        bar_h = int(acc / 100 * 250)
        draw.rectangle([x, bottom_y - bar_h, x + bar_w, bottom_y], fill=color)
        # 标签
        draw.text((x + 8, bottom_y + 10), f'手势{i}', fill='#333', font=font_small)
        # 数值
        draw.text((x + 5, bottom_y - bar_h - 22), f'{acc:.1f}%', fill=color, font=font_normal)

    # Y轴
    for val in [0, 25, 50, 75, 100]:
        y = bottom_y - int(val / 100 * 250)
        draw.text((40, y - 8), f'{val}%', fill='#888', font=font_tiny)

    img.save(os.path.join(SAVE_DIR, 'class_accuracy.png'))
    print('图3: class_accuracy.png 完成')

create_confusion_matrix()
create_training_curve()
create_class_bar()

print(f'\n所有图片保存在: {SAVE_DIR}')
for f in sorted(os.listdir(SAVE_DIR)):
    size = os.path.getsize(os.path.join(SAVE_DIR, f)) // 1024
    print(f'  {f} ({size} KB)')
