# base 基类文档

本模块包含神经网络核心组件的基础类，供所有具体层（如卷积层、线性层、激活层等）继承使用。其目的是规范模块结构，统一前向传播接口，并管理可学习参数。

---

## 1. Layer 基类

`Layer` 是所有神经网络层的基类（linear、convolution、ReLU 等），提供了通用接口和基本属性，简化派生类的实现。

### 1.1. 方法简介

1. 构造函数 `__init__`
- 初始化 `_params` 属性为一个空列表，用于收集该层的所有可学习参数（`Parameter` 对象）。
- 设置 `training` 标志，表示当前是否处于训练模式。

---

2. 属性 `parameters`
- 返回当前层中所有参数组成的列表。
- 用于模型训练或参数更新时统一访问所有参数。

---

3. 重载 `__setattr__`
- 拦截属性设置过程，如果赋值的是 `Parameter` 对象，则将其自动添加到 `_params` 中。
- 实现自动参数收集机制，方便模型构建与优化。

---

4. 前向传播接口 `__call__` 与 `forward`
- `__call__` 方法使 `Layer` 对象可像函数一样被调用，自动转发到 `forward`。
- `forward` 调用 `_forward`，为子类预留实际计算过程的实现入口。

---

5. 抽象前向函数 `_forward`
- 抽象方法，必须在子类中实现，用于定义该层的实际前向传播逻辑。
- 参数 `X` 是输入张量，返回值是输出张量。

---

6. 梯度归零 `zero_grad`
- 将该层中所有参数的梯度置零，调用 `Parameter` 的 `zero_grad` 方法。
- 在每轮反向传播前应调用，以防止梯度累积。

---

7. 模式切换 `train` 和 `eval`
- 切换训练/评估模式。某些层（如 Dropout, BatchNorm）在不同模式下行为不同。
- `train()` 设置 `self.training = True`，`eval()` 设置为 `False`。

---

### 1.3. 使用说明

自定义层时应继承 `Layer` 类并实现 `_forward` 方法。示例：

```python
class MyLayer(Layer):
    def __init__(self):
        super().__init__()
        self.weight = Parameter(...)
        self.bias = Parameter(...)

    def _forward(self, X):
        return X @ self.weight + self.bias
```

该设计使每一层都能自动收集参数、统一调用接口，同时每一层只需实现 `_forward` 方法，接口隔离，模块解耦，并方便地切换训练和评估模式。

---

## 2. Loss 基类

`Loss` 是所有损失函数的基类（如 `CrossEntropyLoss` 等），统一定义了损失计算和反向传播接口，并支持 L1/L2 正则化项。

---

### 2.1. 方法简介

1. 构造函数 `__init__(lambda1=0, lambda2=0)`

* 初始化正则化系数 `lambda1`（L1）和 `lambda2`（L2）。
* 初始化内部属性 `_loss` 为 `None`，用于缓存前向传播结果以便后续反向传播。

---

2. 调用接口 `__call__(pred, true, params)`

* 作为标准调用接口，自动调用 `forward(pred, true, params)`。
* `pred` 是模型预测值，`true` 是真实标签，`params` 是模型所有参数，用于计算正则项。

---

3. 前向传播 `forward(pred, true, params)`

* 实际执行损失函数的前向传播并添加正则化项。
* 自动将 `pred` 和 `true` 转换为 `Tensor` 类型（若尚未为 `Tensor`）。
* 正则项计算如下：

  * **L2 正则化**：对所有启用正则的参数计算平方和；
  * **L1 正则化**：对所有启用正则的参数计算绝对值和。
* 返回主损失值（未加正则项），但 `_loss` 内部保存的是含正则的总损失，用于反向传播。

---

4. 抽象方法 `_forward(pred, true)`

* 子类需实现该方法，用于定义具体的损失函数形式（如均方误差、交叉熵等）。
* 不含正则项逻辑，仅处理核心损失值计算。

---

5. 反向传播 `backward()`

* 对 `_loss`（正则项 + 主损失）执行反向传播。
* 要求 `_loss` 必须是一个标量 `Tensor`，否则抛出断言错误。
* 每次调用后将 `_loss` 重置为 `None`，避免重复传播。

---

### 2.2. 使用说明

自定义损失函数时，应继承 `Loss` 类并实现 `_forward(pred, true)` 方法。例如：

```python
class MyLoss(Loss):
    def _forward(self, pred, true):
        return ((pred - true) ** 2).mean()
```

调用时直接传入预测值、标签和参数列表：

```python
loss_fn = MyLoss(lambda1=0.01, lambda2=0.001)
loss = loss_fn(pred, true, model.parameters())
loss.backward()
```

该设计隔离了正则化逻辑与具体损失定义，使得损失函数的实现更简洁、扩展更方便，并支持灵活的正则化策略。

---

## 3. Optimizer 基类

`Optimizer` 是所有优化器的基类（如 `SGD`, `Adam` 等），定义了统一的参数更新接口，简化优化器的实现和调用。

---

### 3.1 方法简介

1. 构造函数 `__init__()`

* 占位初始化函数，无特殊逻辑。

---

2. 调用接口 `__call__(params)`

* 使优化器对象可像函数一样被调用，等价于执行 `step(params)`。

---

3. 参数更新 `step(params)`

* 调用 `_step(params)` 执行具体的优化策略。
* `params` 为模型中所有 `Parameter` 组成的列表。

---

4. 抽象方法 `_step(params)`

* 子类必须实现的核心优化逻辑。
* 通常对每个参数根据其 `.grad` 执行更新。

---

### 3.2 使用说明

自定义优化器时应继承 `Optimizer` 类并实现 `_step` 方法。例如：

```python
class SGD(Optimizer):
    def __init__(self, lr=0.01):
        self.lr = lr

    def _step(self, params):
        for param in params:
            delta_grad = self._lr * param.grad
            param.step(delta_grad)
```

调用方式如下：

```python
opt = SGD(lr=0.01)
opt(model.parameters())
```

该设计统一了调用接口，便于灵活扩展不同类型的优化器。


