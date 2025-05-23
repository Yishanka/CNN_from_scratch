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
<!-- | `activation/` | 神经网络激活层模块，如 `ReLU`, `SoftMax` 等，继承自 `Layer` | -->
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
### 两种模型初始化方法

#### 动态初始化模型
```python
import cnn
from cnn.layer import Layer
class SimpleCNN(cnn.Model)
  def __init__(self):
      super().__init__()
      self.layer1 = Layer(...)
      self.layer2 = Layer(...)
      self.
model = SimpleCNN()
```
#### 静态初始化
```python
import cnn
from cnn.layer import Layer
layer1 = Layer(...)
layer2 = Layer(...)
...
model = cnn.Model(layer1, layer2, ...)
```

#### 实际实现时，模型初始化用第一种方法，并sequential兼容第二种方法
```python
import cnn
from cnn.layer import Layer
layer1 = Layer(...)
layer2 = Layer(...)
...
model = cnn.Model()
model.sequential(layer1, layer2, ...)
```
### 模型预测和训练过程
#### 自己定义，灵活性高
```python
# 假设 x, true 为训练数据
pred = model.forward(x)
loss = model.compute_loss(pred, true)
model.backward()
model.step()
model.zero_grad()
```

#### 用集成的fit方法

---

## 开发建议

1. **模块独立：** 每个包内部尽量保持职责单一，避免耦合。每个类尽量从 `base` 中的基类继承，保证功能完整。
2. **注释规范：** 所有公开方法添加参数说明、返回说明与功能注释。
3. **文档维护：** 每个模块建议附带 `README.md` 说明其功能与开发进度。
4. **测试结构：** 所有测试代码统一由开发者在本地根目录下创建的 `test_*.py` 文件中，模拟真实调用场景；`test.py` 不会上传到 GitHub。

---

## 开发注意事项：
1. 整体的数据流的维度是：
    - X: (batch_size, feature_size_1, feature_size_2, ......)
        - eg: X = \[[1, 2], [1, 2], [1, 2]]: batch_size = 3, feature_size_1 = 2
    - y (regression): (batch_size, 1)
        - eg: y = \[[0], [1], [2]]: batch_size = 3（列向量）
    - y (classification): (batch_size)
        - eg: y = [0, 1, 2]: batch_size = 3（行向量）
        - *暂时未实现独热编码，后续实现后可能需要修改*
2. 层接受输入的维度需符合上述要求，参数和计算的设计要保证输出维度也需符合上述要求
3. 私有属性和方法在变量名前加单下划线 (_)，尽量不要让派生类获取父类的私有属性

## 附录：计算速度优化方法
1. 反向传播时先对计算图剪枝：
    - 删除所有不需要求导的结点
    - 原因：不需要求导的结点的子结点也不需要求导，则该子图上所有结点都不需要求导

2. 设置模型 `train`/`eval` 模式选择
    - 在验证/测试时避免反复构建+删除计算图

3. `__getitem__ ` 不再创建临时变量 `grad`，避免临时分配内存，性能大幅优化

4. `as_strided` 不重新分配内存，加快 `im2col` 操作

5. 原地操作的使用，减少结点创建
    - 使用时要保证单向数据流，不能有右值继续计算的情况
    - 没有反向传播时（测试），模型参数即其他永久变量的视图被永久修改，计算错误，待解决

6. 反向传播临时变量预计算：将与父结点传回梯度无关的计算和变量放在 `_backward` 函数体外，避免重复计算

7. `np.add` 原地计算，尽量避免临时内存创建

8. 使用 `einsum` 替代手动降维，更高效且易读