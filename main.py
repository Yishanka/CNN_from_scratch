import numpy as np

import cnn
from cnn.core import Tensor, LossMonitor, MetricMonitor
from cnn.layer import Linear, ReLU, LeakyReLU, Conv2d, Flatten, MaxPool2d, Softmax, BatchNorm2d
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

model = cnn.Model()
model.sequential(
    # block 1
    Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1),
    ReLU(),
    # BatchNorm2d(channels=8),
    MaxPool2d(kernel_size=2),

    # block 2
    Conv2d(in_channels=8, out_channels=32, kernel_size=3, stride=1, padding=1),
    ReLU(),
    # BatchNorm2d(channels=32),
    MaxPool2d(kernel_size=2),
 
    # flatten and dense
    Flatten(),
    Linear(in_features=32*7*7, out_features=128),
    ReLU(),

    # output
    Linear(in_features=128, out_features=10),
    Softmax()
)

model.compile(
    loss=CrossEntropyLoss(lambda2=0.05),
    optimizer = Adam(lr=0.00001)
)

# <=== 训练 ===>
loss_monitor = LossMonitor(STOP_KEY) 
for epoch in range(5):
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





