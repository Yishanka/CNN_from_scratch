# Convolution Neural Network from Scratch!
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/Yishanka/CNN_from_scratch)

## 项目简介
这是一个 **从零实现的卷积神经网络（CNN）框架**，仅依赖 `numpy`，目标是构建一个结构清晰、模块解耦、具有可拓展性的深度学习系统。目前支持图像分类任务，并预留接口以支持更多模型与功能扩展。

---

## 核心理念

本项目**以参数为中心**，明确区分模型定义与训练逻辑，各模块职责单一、可插拔。训练流程通过 Model 类集中调度，保持上层调用的整洁性与可控性。

---

## 架构设计

### 核心组件（内部使用）
| 模块名 | 说明 |
|--------|------|
| `core/` | 核心张量计算模块，实现 `Tensor` 与其子类 `Parameter`，支持基本运算与自动求导 |
| `base/` | 所有基类，如 `Layer`, `Loss`, `Optimizer` 等，定义组件交互协议 |

### 模型组件（对外接口）
| 模块名 | 说明 |
|--------|------|
| `layer/` | 神经网络层模块，如 `Linear`, `Conv2d` 等，继承自 `Layer` |
| `activation/` | 神经网络激活层模块，如 `ReLU`, `SoftMax` 等，继承自 `Layer` |
| `loss/` | 损失函数模块，如 `CrossEntropyLoss`，继承自 `Loss` |
| `optimizer/` | 优化器模块，如 `Adam`，继承自 `Optimizer` |
| `data/` | 数据处理模块，负责数据加载、预处理等 |

---

## 模块职责说明

### Tensor 与 Parameter（`core/`）
- `Tensor`: 存储数据及其梯度，支持前向计算与反向传播。
- `Parameter`: 继承自 Tensor，标记为需要被优化的变量（即模型的“学习参数”）。

### Layer（`base/Layer`）
- 所有模型层的基类，定义 forward 接口。
- 每一层持有自己的 Parameter （激活层不持有参数），不直接关心优化与损失。
- 执行 `zero_grad()` 将参数梯度设为 0

### Loss（`base/Loss`）
- 定义损失函数的 forward 计算。
- 内部维护 loss 变量，并提供 `backward()` 方法，控制整个计算图的反向传播过程。

### Optimizer（`base/Optimizer`）
- 提取所有 Parameter。
- 执行 `step()` 时更新参数。

### Model（`base/Model`）
- 封装所有组件（Layer、Loss、Optimizer）
- 定义训练接口：`forward`, `compute_loss`, `backward`, `step`, `zero_grad`。
- 实例化模型时只需将 Layer、Loss、Optimizer 赋值为属性即可被自动注册。

---

## 使用示例
```python
import numpy as np
import matplotlib.pyplot as plt

import cnn
from cnn.core import Tensor
from cnn.layer import Linear, ReLU, Conv2d, Flatten, MaxPool2d, Softmax, BatchNorm2d
from cnn.optimizer import Adam
from cnn.loss import CrossEntropyLoss

model = cnn.Model()
model.sequential(
    Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
    BatchNorm2d(channels=16),
    ReLU(),
    MaxPool2d(kernel_size=2),

    Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=1, padding=1),
    BatchNorm2d(channels=64),
    ReLU(),
    MaxPool2d(kernel_size=2),

    Flatten(),

    Linear(in_features=64*7*7, out_features=128),
    ReLU(),
    Linear(in_features=128, out_features=10),
    Softmax()
)

model.compile(
    loss=CrossEntropyLoss(lambda2=0.02),
    optimizer = Adam(lr=1e-4, beta1=0.9, beta2=0.999)
)

# 训练
for epoch in range(5):
    model.train() # 必须执行，保证参数参与计算图构建
    for (X, y) in enumerate(train_loader):
        pred = model.forward(X) 
        loss = model.loss(pred, y)
        model.backward(remove_graph=True)
        model.step()
        model.zero_grad()

# 训练集上测试
model.eval() # 必须执行，保证参数不参与计算图构建
for X, y in train_loader:
    pred = model.forward(X)
    loss = model.loss(pred, y)

# 测试集上测试
for X, y in test_loader:
    pred = model.forward(X)
    loss = model.loss(pred, y)
```
可根据需求，自定义指标计算等，暂不集成到框架中。

完整的代码可参考 `main.py`，直接运行可以对 `FashionMNIST` 数据集做分类，在给定神经网络框架下一轮大约需要 7 分钟。

启动 `monitor.py`，可以看到分类器在 `FashionMNIST` 一轮训练种损失下降的曲线。

可以通过其他 dataloader 加载其他数据集进行学习与分类。

---
## 下一步
1. 可视化
    - 卷积层输出特征的可视化
    - 张量类生成的计算图可视化
    - 交互前端设计

2. 模型参数的保存与加载