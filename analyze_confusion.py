import sys; sys.path.insert(0, '.')
import torch
import numpy as np
from model import CustomNet
from dataset import CustomDataset
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

model = torch.load('./models/model.pkl', map_location='cpu', weights_only=False)
model.eval()

ds = CustomDataset('./images/test.txt', './images/test', ToTensor)
dl = DataLoader(ds, batch_size=32)

correct = {i:0 for i in range(6)}
total = {i:0 for i in range(6)}
confusion = np.zeros((6,6), dtype=int)

with torch.no_grad():
    for batch in dl:
        x, y = batch['image'], batch['label']
        out = model(x)
        pred = out.argmax(dim=1)
        for t, p in zip(y.tolist(), pred.tolist()):
            confusion[t][p] += 1
            total[t] += 1
            if t == p: correct[t] += 1

print('=== 各类别测试准确率 ===')
for i in range(6):
    print(f'数字 {i}: {correct[i]}/{total[i]} = {100*correct[i]/total[i]:.1f}%')

print()
print('=== 混淆矩阵 (行=真实, 列=预测) ===')
print('      ', end='')
for j in range(6): print(f' 预{j}  ', end='')
print()
for i in range(6):
    print(f'真{i}  ', end='')
    for j in range(6):
        print(f' {confusion[i][j]:3d} ', end='')
    print()

print()
mis = [(i,j) for i in range(6) for j in range(6) if i!=j and confusion[i][j]>0]
for i,j in mis:
    print(f'  真实{i} 误判为 {j}: {confusion[i][j]}次')
