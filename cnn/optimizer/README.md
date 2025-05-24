# Optimizer 模块文档

## 1. Adam 优化器（Adaptive Moment Estimation）

---

### 1. 概念简述

**Adam（Adaptive Moment Estimation）优化器**是一种结合了动量法（Momentum）和 RMSProp 优点的优化算法。它利用一阶矩（梯度的指数移动平均）和二阶矩（梯度平方的指数移动平均）动态调整每个参数的学习率，从而提升模型训练的收敛速度与稳定性。

Adam 是目前应用最广泛的优化器之一，尤其在训练大型神经网络（如 CNN、RNN）时表现优异。

---

### 2. 数学原理

Adam 在每一步更新时，依赖以下几个核心变量：

* \$ g\_t \$：当前时刻 \$ t \$ 的参数梯度；
* \$ m\_t \$：梯度的一阶矩估计（动量项）；
* \$ v\_t \$：梯度的二阶矩估计（方差项）；
* \$ \hat{m}\_t, \hat{v}\_t \$：对 \$ m\_t, v\_t \$ 的偏差修正（bias correction）；
* \$ \beta\_1, \beta\_2 \$：控制一阶、二阶矩估计的衰减率；
* \$ \epsilon \$：数值稳定项，避免除零。

参数更新公式如下：

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t \\
v_t &= \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \\
\hat{m}_t &= \frac{m_t}{1 - \beta_1^t} \quad \text{(偏差修正)} \\
\hat{v}_t &= \frac{v_t}{1 - \beta_2^t} \\
\theta_t &= \theta_{t-1} - \alpha \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
\end{aligned}
$$

其中 \$\theta\_t\$ 表示当前参数，\$\alpha\$ 为全局学习率。

---

### 3. 代码实现重点解析

```python
def _step(self, params: list[Parameter], lr):
    self._t += 1  # 时间步增加，用于偏差修正
    for i, param in enumerate(params):
        if i not in self.m:
            self.m[i] = Tensor.zeros_like(param)
            self.v[i] = Tensor.zeros_like(param)

        # 更新一阶与二阶动量估计
        self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * param.grad
        self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (param.grad ** 2)

        # 偏差修正
        m_hat = self.m[i] / (1 - self.beta1 ** self._t)
        v_hat = self.v[i] / (1 - self.beta2 ** self._t)

        # 计算更新步长
        delta_grad = lr * m_hat / (v_hat**0.5 + self.eps)

        # 更新参数（调用 param 自带的原地更新方法）
        param.step(delta_grad)
```

**实现要点：**

* 使用哈希表（dict）记录每个参数的动量状态，支持参数数量动态变化。
* 使用 `Tensor.zeros_like(param)` 创建与参数形状一致的动量张量。
* 每步更新前递增 `self._t`，确保偏差修正项有效。
* 由于 `param.step()` 是原地更新，不涉及新张量构建，因此兼容用户自定义的轻量级 `Tensor` 实现。

---

### 4. 参数说明

| 参数             | 作用说明                 |
| -------------- | -------------------- |
| `lr`           | 全局学习率（初始步长）          |
| `min_lr`       | 最小学习率下限，配合 decay 使用  |
| `beta1`        | 一阶动量衰减系数，控制历史梯度的保留程度 |
| `beta2`        | 二阶动量衰减系数，控制梯度方差的平滑程度 |
| `eps`          | 数值稳定项，避免分母为零         |
| `lr_decay` | 每步学习率衰减系数，控制 lr 动态变化 |

---

### 5. 使用建议与注意事项

* **推荐默认超参数**：`beta1 = 0.9, beta2 = 0.999, eps = 1e-8` 是深度学习中普遍使用的默认配置。
* **不建议搭配较大的 batch size** 使用较高的 `lr`，以免引起梯度爆炸。
* **应搭配适当的正则化策略**以增强泛化能力。

---

### 6. 可拓展方向

* **加入 `weight_decay` 参数**，以支持 L2 正则项；
* **支持 `amsgrad` 版本**（保存最大历史二阶矩）增强收敛稳定性；
* **动态调整 lr 策略**（如 warmup、cosine decay）；
* **梯度裁剪（gradient clipping）** 支持，防止梯度爆炸；
* 将 `m` 与 `v` 缓存转为 `Parameter` 属性，支持断点训练与状态恢复。
