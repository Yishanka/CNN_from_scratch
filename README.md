# Convolution Neural Network from Scratch!

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
import cnn
from cnn.data import loader
from cnn.layer import Linear
from cnn.optimizer import Adam
from cnn.loss import CrossEntropyLoss

class TestModel(cnn.Model):
    def __init__(self):
        super().__init__()
        self.fc = Linear(2, 1)
        self.optimizer = Adam()
        self.loss = CrossEntropyLoss()

# 构造模型
model = TestModel()

# 假设 x, true 为训练数据
pred = model.forward(x)
loss = model.compute_loss(pred, true)
model.backward()
model.step()
model.zero_grad()
```

---

## 开发建议

1. **模块独立：** 每个包内部尽量保持职责单一，避免耦合。每个类尽量从 `base` 中的基类继承，保证功能完整。
2. **注释规范：** 所有公开方法添加参数说明、返回说明与功能注释。
3. **文档维护：** 每个模块建议附带 `README.md` 说明其功能与开发进度。
4. **测试结构：** 所有测试代码统一由开发者在本地根目录下创建的 `test_*.py` 文件中，模拟真实调用场景；`test.py` 不会上传到 GitHub。

---

## 注意事项：
1. 整体的数据流的维度是：
    - X: (batch_size, feature_size_1, feature_size_2, ......)
        - eg: X = \[[1, 2], [1, 2], [1, 2]]: batch_size = 3, feature_size_1 = 2
    - y (regression): (batch_size, 1)
        - eg: y = \[[0], [1], [2]]: batch_size = 3（列向量）
    - y (classification): (batch_size)
        - eg: y = [0, 1, 2]: batch_size = 3（行向量）
        - *暂时未实现独热编码，后续实现后可能需要修改*
2. 层接受输入的维度需符合上述要求，参数和计算的设计要保证输出维度也需符合上述要求
