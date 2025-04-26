import numpy as np
import cnn
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

# <=== 训练 ===>
train_dataset = FashionMNIST(root='./data', train=True)
train_dataset.to_one_hot()
train_loader = DataLoader(train_dataset.get_data(), batch_size=64, shuffle=True)
X_whole, y_whole = train_dataset.get_data()

class SimpleCNN(cnn.Model):
    def __init__(self):
        super().__init__()
        # [64, 1, 28, 28]
        self.conv1 = Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        # [64, 32, 28, 28],
        self.ac1 = ReLU()
        # [64, 32, 28, 28]
        self.pool1 = MaxPool2d(kernel_size=4)
        # [64, 32, 7, 7]
        self.flatten1 = Flatten()
        # [64, 32*7*7]
        self.fc1 = Linear(in_features=32*7*7, out_features=128)
        self.ac3 = ReLU()
        self.fc2 = Linear(in_features=128, out_features=10)
        self.ac4 = Softmax()
        self.optimizer = Adam(lr=0.0001)
        self.loss = CrossEntropyLoss(lambda2=1)

pred = None
loss = None
model = SimpleCNN()
# early stopping
i = 0
for epoch in range(5):
    for X, y in train_loader:
        pred = model.forward(X)
        loss = model.compute_loss(pred, y)
        model.backward()
        model.step()
        model.zero_grad()
    print(loss)
    if loss._data<0.4:
        break

# <=== 测试 ===>
test_dataset = FashionMNIST(root='./data', train=False)
test_dataset.to_one_hot()
test_loader = DataLoader(test_dataset.get_data(), batch_size=64, shuffle=False)

correct = 0
total = 0

for X, y in test_loader:
    pred = model.forward(X)
    pred_labels = np.argmax(pred._data, axis=1)
    true_labels = np.argmax(y._data, axis=1)
    correct += np.sum(pred_labels == true_labels)
    total += len(true_labels)

accuracy = correct / total
print(f"Test Accuracy: {accuracy:.4f}")