import numpy as np


import cnn
from cnn.core import Tensor, LossMonitor, MetricMonitor
from cnn.layer import Linear, ReLU, LeakyReLU, Conv2d, Flatten, MaxPool2d, Softmax
from cnn.optimizer import SGD, Adam
from cnn.loss import MSELoss, CrossEntropyLoss
from cnn.data import FashionMNIST, DataLoader

# 检查内存与对象数量
# import os, psutil
# import gc
# process = psutil.Process(os.getpid())
# print("Memory:", process.memory_info().rss / 1024**2, "MB")
# print(f"Objects alive: {len(gc.get_objects())}")

# 常量
STOP_KEY = 'space'  # 按 's' 键停止训练

# <=== 训练数据集 ===>
train_dataset = FashionMNIST(root='./data', train=True)
train_dataset.to_one_hot()
train_loader = DataLoader(train_dataset.get_data(), batch_size=128, shuffle=True)
X_whole, y_whole = train_dataset.get_data()

# <=== 测试 ===>
test_dataset = FashionMNIST(root='./data', train=False)
test_dataset.to_one_hot()
test_loader = DataLoader(test_dataset.get_data(), batch_size=64, shuffle=False)

class SimpleCNN(cnn.Model):
    def __init__(self):
        super().__init__()
        # [64, 1, 28, 28]
        self.conv1 = Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1)
        # [64, 8, 28, 28],
        self.ac1 = ReLU()
        # [64, 8, 28, 28]
        self.pool1 = MaxPool2d(kernel_size=2)
        # [64, 8, 14, 14]
        self.conv2 = Conv2d(in_channels=8, out_channels=32, kernel_size=3, stride=1, padding=1)
        # [64, 32, 14, 14],
        self.ac2 = ReLU()
        # [64, 32, 14, 14]
        self.pool2 = MaxPool2d(kernel_size=2)
        # [64, 32, 7, 7]
        self.flatten1 = Flatten()
        # [64, 32*7*7]
        self.fc1 = Linear(in_features=32*7*7, out_features=128)
        self.ac3 = ReLU()
        self.fc2 = Linear(in_features=128, out_features=10)
        self.ac4 = Softmax()
        self.optimizer = Adam(lr=0.0003)
        self.loss = CrossEntropyLoss(lambda2=0.7)

model = SimpleCNN()
# <=== 训练 ===>
loss_monitor = LossMonitor(STOP_KEY) 
for epoch in range(4):
    if not loss_monitor.is_training:
        break
    
    for X, y in train_loader:
        if not loss_monitor.is_training:
            break

        pred = model.forward(X)
        loss = model.compute_loss(pred, y)
        model.backward()
        model.step()
        model.zero_grad()

        # 记录 loss
        loss_monitor.append_loss(loss=loss)
        loss_monitor.update_plots()

# <=== 测试 ===> 
accuracy_moniter = MetricMonitor(type='accuracy')
correct = 0
total = 0
for X, y in test_loader:
    pred = model.forward(X)
    pred_labels = Tensor.argmax(pred, axis=1)
    true_labels = Tensor.argmax(y, axis=1)
    correct += np.sum(pred_labels == true_labels)
    total += len(true_labels)
    accuracy_moniter.append_metric(correct / total)
    accuracy_moniter.update_plots()

accuracy = correct / total

# monitor.accuracies.append(accuracy)
print(f"Epoch {epoch + 1}, Test Accuracy: {accuracy:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")





