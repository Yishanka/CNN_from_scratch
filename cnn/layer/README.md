# layer 模块文档
## 全连接层（Linear）

线性层实现经典的仿射变换：  
$$
\text{output} = XW + b
$$  
其中：
- $ X $ 是输入张量，形状为 `(batch_size, in_features)`
- $ W $ 是权重参数，形状为 `(in_features, out_features)`
- $ b $ 是偏置参数，初始形状为 `(1, out_features)`，将自动通过广播扩展到 batch 尺寸

### 参数说明：
| 参数名        | 类型     | 说明                                 |
|---------------|----------|--------------------------------------|
| `in_features` | `int`    | 输入特征维度                         |
| `out_features`| `int`    | 输出特征维度                         |

### 设计说明：

- **参数封装**：权重和偏置均封装为 `Parameter` 对象，支持注册和自动求导。
- **初始化策略**：默认使用 He 初始化（适合 ReLU 激活）。
### 示例代码：

```python
layer = Linear(64, 10)
out = layer(x)  # x: Tensor of shape (batch_size, 64)
```

---
