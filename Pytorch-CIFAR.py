import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
import time
import threading
import keyboard

# 设置matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号

# 1. 损失监控类
class LossMonitor:
    def __init__(self, stop_key: str = 'space'):
        self.losses = []
        self.is_training = True
        self.stop_key = stop_key
        
        stop_thread = threading.Thread(target=self.check_stop_key)
        stop_thread.daemon = True
        stop_thread.start()
    
    def append_loss(self, loss: float):
        self.losses.append(loss)
    
    def update_plots(self):
        pass  # 不绘制损失曲线
    
    def check_stop_key(self):
        while self.is_training:
            if keyboard.is_pressed(self.stop_key):
                self.stop_training()
                print("训练已被用户中断")
                time.sleep(2)  # 防止重复检测
            time.sleep(0.1)
    
    def stop_training(self):
        self.is_training = False

# 2. 数据加载
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 标准化RGB图像
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# 类别标签
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 3. 定义模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 卷积层1: 输入3通道
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding='same')  # 输出尺寸: (32, 32, 32)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 输出尺寸: (32, 16, 16)
        
        # 卷积层2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding='same')  # 输出尺寸: (64, 16, 16)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 输出尺寸: (64, 8, 8)
        
        # 卷积层3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding='same')  # 输出尺寸: (128, 8, 8)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 输出尺寸: (128, 4, 4)
        
        # 全连接层
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.dropout = nn.Dropout(0.5)  # 添加Dropout防止过拟合
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        # 第一层卷积块
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.pool1(x)
        
        # 第二层卷积块
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.pool2(x)
        
        # 第三层卷积块
        x = self.conv3(x)
        x = self.bn3(x)
        x = torch.relu(x)
        x = self.pool3(x)
        
        # 全连接层
        x = torch.flatten(x, 1)  # 保持batch维度
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)  # 应用Dropout
        x = self.fc2(x)
        return x

# 进度条函数
def print_progress(iteration, total, prefix='', suffix='', length=50, fill='█'):
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='\r')
    if iteration == total:
        print()

# 4. 训练与验证
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

# 存储指标的列表
train_accs, test_accs = [], []
precisions, recalls, f1s = [], [], []

# 初始化损失监控器
loss_monitor = LossMonitor(stop_key='space')

print("开始训练... (按空格键可随时停止)")

total_epochs = 10
for epoch in range(total_epochs):  # 训练轮次
    if not loss_monitor.is_training:
        print("\n训练提前终止")
        break
        
    print(f"\nEpoch {epoch+1}/{total_epochs}:")
    
    # 训练阶段
    model.train()
    correct, total = 0, 0
    train_loss = 0
    total_batches = len(train_loader)
    
    print("训练中: ", end='')
    for batch_idx, (data, target) in enumerate(train_loader):
        if not loss_monitor.is_training:
            break
            
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        loss_value = loss.item()
        loss_monitor.append_loss(loss_value)
        train_loss += loss_value
        
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
        
        # 更新进度条
        print_progress(batch_idx + 1, total_batches, prefix='', suffix='完成')
        
        # 检查是否需要停止训练
        if not loss_monitor.is_training:
            break
    
    if not loss_monitor.is_training:
        break
        
    train_acc = correct / total
    train_accs.append(train_acc)
    avg_train_loss = train_loss / total_batches
    
    # 验证阶段
    model.eval()
    correct, total = 0, 0
    all_preds = []
    all_targets = []
    total_batches = len(test_loader)
    
    print("验证中: ", end='')
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            
            # 更新进度条
            print_progress(batch_idx + 1, total_batches, prefix='', suffix='完成')
    
    test_acc = correct / total
    test_accs.append(test_acc)
    
    # 计算精确率、召回率、F1分数（macro平均）
    precision = precision_score(all_targets, all_preds, average='macro')
    recall = recall_score(all_targets, all_preds, average='macro')
    f1 = f1_score(all_targets, all_preds, average='macro')
    precisions.append(precision)
    recalls.append(recall)
    f1s.append(f1)
    
    # 打印本轮完整指标
    print(f"训练准确率={train_acc:.4f}, 测试准确率={test_acc:.4f}, "
          f"精确率={precision:.4f}, 召回率={recall:.4f}, F1分数={f1:.4f}")

# 5. 训练完成后绘制综合指标图
plt.figure(figsize=(12, 6))
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
lines = []
labels = ['train_accuracy', 'test_accuracy', 'precision', 'recall', 'f1']
for key, data, color in zip(labels, [train_accs, test_accs, precisions, recalls, f1s], colors):
    line, = plt.plot(data, label=key, color=color)
    lines.append(line)

plt.xlabel("Epoch")
plt.ylabel("Metric")
plt.title("Metrics per Epoch")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()