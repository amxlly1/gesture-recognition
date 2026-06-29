import requests
import numpy as np
from PIL import Image, ImageDraw
import io

# 测试1: 手势图（手在中央）→ 应该检测到手并识别
print("=== 测试 1: 标准手势图 ===")
f = open('./images/test/signs/img_0006.png', 'rb')
r = requests.post('http://127.0.0.1:5000/predict', files={'image':('t.png',f,'image/png')})
d = r.json()
print(f"  pred={d['prediction']} conf={d['confidence']}% gesture={d['is_gesture']} hand={d['hand_detected']}")

# 测试2: 纯色图（无手）→ 应该拒识
print("=== 测试 2: 纯色背景 ===")
img = Image.new('RGB', (200, 200), color=(120, 120, 120))
buf = io.BytesIO(); img.save(buf, 'PNG'); buf.seek(0)
r = requests.post('http://127.0.0.1:5000/predict', files={'image':('t.png',buf,'image/png')})
d = r.json()
print(f"  pred={d['prediction']} conf={d['confidence']}% gesture={d['is_gesture']} hand={d['hand_detected']}")

# 测试3: 手在画面角落（模拟你拍的图）→ 应该检测到手部ROI
print("=== 测试 3: 手在角落 ===")
# 创建一张模拟图：肤色区域在角落
img = Image.new('RGB', (400, 300), color=(60, 60, 80))
draw = ImageDraw.Draw(img)
# 在左上角画一个肤色椭圆（模拟手）
draw.ellipse([30, 20, 150, 150], fill=(210, 170, 140))
draw.ellipse([50, 40, 100, 90], fill=(200, 160, 130))  # 手指简化
buf = io.BytesIO(); img.save(buf, 'PNG'); buf.seek(0)
r = requests.post('http://127.0.0.1:5000/predict', files={'image':('t.png',buf,'image/png')})
d = r.json()
print(f"  pred={d['prediction']} conf={d['confidence']}% gesture={d['is_gesture']} hand={d['hand_detected']} bbox={d.get('bbox')}")

print("\n=== 完成 ===")
