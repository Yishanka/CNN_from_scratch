# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

# 设置matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号

# 1. 数据加载
transform = transforms.ToTensor()
train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# 2. 定义模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(64 * 24 * 24, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.max_pool2d(torch.relu(torch.batch_norm(self.conv1(x))))
        x = torch.max_pool2d(torch.relu(torch.batch_norm(self.conv2(x))))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 3. 训练与验证
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

train_losses, test_losses = [], []
train_accs, test_accs = [], []
precisions, recalls, f1s = [], [], []

for epoch in range(10):
    # 训练
    model.train()
    correct, total, train_loss = 0, 0, 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
    train_losses.append(train_loss / len(train_loader))
    train_accs.append(correct / total)

    # 验证
    model.eval()
    correct, total, test_loss = 0, 0, 0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    test_losses.append(test_loss / len(test_loader))
    test_accs.append(correct / total)

    # 计算精确率、召回率、F1分数
    precision = precision_score(all_targets, all_preds, average='macro')
    recall = recall_score(all_targets, all_preds, average='macro')
    f1 = f1_score(all_targets, all_preds, average='macro')
    precisions.append(precision)
    recalls.append(recall)
    f1s.append(f1)

    print(f"第{epoch+1}轮: 训练损失={train_losses[-1]:.4f}, 测试损失={test_losses[-1]:.4f}, "
          f"训练准确率={train_accs[-1]:.4f}, 测试准确率={test_accs[-1]:.4f}, "
          f"精确率={precision:.4f}, 召回率={recall:.4f}, F1分数={f1:.4f}")

# 4. 可视化
plt.figure(figsize=(18,5))

plt.subplot(1,3,1)
plt.plot(train_losses, label='训练损失')
plt.plot(test_losses, label='测试损失')
plt.legend()
plt.title('损失曲线')

plt.subplot(1,3,2)
plt.plot(train_accs, label='训练准确率')
plt.plot(test_accs, label='测试准确率')
plt.legend()
plt.title('准确率曲线')

plt.subplot(1,3,3)
plt.plot(precisions, label='精确率')
plt.plot(recalls, label='召回率')
plt.plot(f1s, label='F1分数')
plt.legend()
plt.title('精确率/召回率/F1分数')

plt.tight_layout()
plt.show()