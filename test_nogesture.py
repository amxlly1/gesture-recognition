import requests
import numpy as np
from PIL import Image, ImageFilter
import io

# 模拟各类非手势场景
tests = []

# 纯色背景
for color in [(50,50,50), (120,120,120), (200,200,200), (80,120,200), (150,80,40)]:
    img = Image.new('RGB', (64,64), color=color)
    buf = io.BytesIO(); img.save(buf, 'PNG'); buf.seek(0)
    tests.append((f'纯色{color}', buf))

# 噪声
arr = np.random.randint(0, 255, (64,64,3), dtype=np.uint8)
img = Image.fromarray(arr)
buf = io.BytesIO(); img.save(buf, 'PNG'); buf.seek(0)
tests.append(('噪声', buf))

# 模糊噪声
arr = np.random.randint(50, 200, (64,64,3), dtype=np.uint8)
img = Image.fromarray(arr).filter(ImageFilter.GaussianBlur(3))
buf = io.BytesIO(); img.save(buf, 'PNG'); buf.seek(0)
tests.append(('模糊噪声', buf))

for name, buf in tests:
    r = requests.post('http://127.0.0.1:5000/predict', files={'image':('t.png',buf,'image/png')})
    d = r.json()
    stat = '✅ 正确拒识' if not d['is_gesture'] else '❌ 误判为手势'
    print(f"{name}: pred={d['prediction']} conf={d['confidence']:.1f}% entropy={d['entropy']:.2f} {stat}")
