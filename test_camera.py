import requests
for i in range(6):
    f = open(f'./images/test/signs/img_{i:04d}.png', 'rb')
    r = requests.post('http://127.0.0.1:5000/predict', files={'image':('t.png',f,'image/png')})
    d = r.json()
    print(f"img_{i:04d}: pred={d['prediction']} conf={d['confidence']}% gesture={d['is_gesture']}")
