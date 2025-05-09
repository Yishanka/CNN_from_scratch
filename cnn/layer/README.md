# layer 模块文档
## 1. 全连接层（Linear）
### 1.1 基本概念
线性层实现经典的仿射变换：  
$$
\text{output} = XW + b
$$  
其中：
- $ X $ 是输入张量，形状为 `(batch_size, in_features)`
- $ W $ 是权重参数，形状为 `(in_features, out_features)`
- $ b $ 是偏置参数，初始形状为 `(1, out_features)`，将自动通过广播扩展到 batch 尺寸

### 1.2 参数说明：
| 参数名        | 类型     | 说明                                 |
|---------------|----------|--------------------------------------|
| `in_features` | `int`    | 输入特征维度                         |
| `out_features`| `int`    | 输出特征维度                         |

### 1.3 设计说明：

- **参数封装**：权重和偏置均封装为 `Parameter` 对象，支持注册和自动求导。
- **初始化策略**：默认使用 He 初始化（适合 ReLU 激活）。
### 1.4 示例代码：

```python
layer = Linear(64, 10)
out = layer(x)  # x: Tensor of shape (batch_size, 64)
```

---


## 2. 卷积层（Conv2D）

### 2.1 基本概念

**卷积操作（Convolution）**是深度学习中提取局部空间特征的核心操作。它通过一个小窗口（称为**卷积核 kernel**）在输入特征图上滑动，对每个区域执行逐元素乘法并求和，生成一个新的输出特征图。

为了高效实现卷积，我们使用 **im2col 展开技巧**，将每个局部区域拉平为一行，然后将整个卷积转化为一个大规模矩阵乘法，从而加速计算。

---

### 2.2 数学公式

设：

* 输入特征图 \$X \in \mathbb{R}^{N \times C\_{\text{in}} \times H \times W}\$，其中 \$N\$ 是 batch size；
* 卷积核参数 \$K \in \mathbb{R}^{C\_{\text{out}} \times C\_{\text{in}} \times k\_H \times k\_W}\$；
* 偏置项 \$b \in \mathbb{R}^{C\_{\text{out}}}\$；
* 输出特征图 \$Y \in \mathbb{R}^{N \times C\_{\text{out}} \times H\_{\text{out}} \times W\_{\text{out}}}\$；

则输出中第 \$n\$ 个样本、第 \$c\$ 个通道、位置 \$(i, j)\$ 的值为：

$$
Y_{n, c, i, j} = \sum_{d=1}^{C_{\text{in}}} \sum_{u=1}^{k_H} \sum_{v=1}^{k_W} X_{n, d, i+u, j+v} \cdot K_{c, d, u, v} + b_c
$$

该过程在空间维度上滑动，每次计算一个局部区域的卷积结果。

---

### 2.3 参数说明

| 参数名            | 说明                          |
| -------------- | --------------------------- |
| `in_channels`  | 输入图像的通道数（灰度图为1，RGB图为3）      |
| `out_channels` | 卷积核个数，决定输出特征图的数量            |
| `kernel_size`  | 卷积核的高宽，常为 `(3,3)` 或 `(5,5)` |
| `stride`       | 卷积核的滑动步长，默认是1               |
| `padding`      | 输入边缘的零填充数量，用于控制输出尺寸         |

---

### 2.4 优化细节

#### 2.4.1 im2col 展开机制

为了将卷积操作转化为矩阵乘法，我们采用了 `im2col` 技术：

* 对每个输入图像，将每个滑动窗口区域拉平成一行；
* 所有窗口的展开组合成一个矩阵 `cols`，形状为 `[batch_size, output_height * output_width, kernel_size * in_channels]`；
* 卷积核也展平为矩阵 `W`，形状为 `[kernel_size * in_channels, out_channels]`；
* 最终输出结果为矩阵乘法：`cols @ W`。

#### 示例代码片段：

```python
cols = im2col(x, (kh, kw), (sh, sw), (ph, pw))   # [bs, oh*ow, ic*kh*kw]
weight_flat = self._weight.reshape((oc, -1)).T   # [ic*kh*kw, oc]
out = cols @ weight_flat                         # [bs, oh*ow, oc]
out = out.transpose((0, 2, 1))                   # [bs, oc, oh*ow]
```

然后 reshape 成最终的卷积输出。

---

#### 2.4.2 im2col 实现细节

```python
def im2col(x: Tensor, kernel_size, stride, padding) -> Tensor:
    _, _, h, w = x.shape
    kh, kw = kernel_size
    sh, sw = stride
    ph, pw = padding

    # 填充边界
    if ph > 0 or pw > 0:
        x = x.pad(((0, 0), (0, 0), (ph, ph), (pw, pw)))

    oh = (h + 2 * ph - kh) // sh + 1
    ow = (w + 2 * pw - kw) // sw + 1

    cols = []
    for i in range(oh):
        for j in range(ow):
            patch = x[:, :, i*sh:i*sh+kh, j*sw:j*sw+kw]  # [bs, ic, kh, kw]
            cols.append(patch.reshape((x.shape[0], -1)))  # 展平为 [bs, ic*kh*kw]
    return Tensor.stack(cols, axis=1)  # 输出 [bs, oh*ow, ic*kh*kw]
```

此函数等价于将输入在每个滑窗位置展开为一个行向量。

---

### 2.6 总结

* 卷积操作可以通过 **im2col + 矩阵乘法** 高效实现；
* 输入、权重、偏置等参数均被设计为 `Tensor` 类支持自动求导；
* padding 和 stride 可以灵活控制输出尺寸；
* im2col 技术是自定义框架中提升性能的关键方法。

---

## 3. 池化层（MaxPool2D）

### 3.1 基本概念

**池化（Pooling）** 是卷积神经网络中的一种下采样操作，用于减小空间尺寸（Height, Width），从而：

* 减少计算量与参数量；
* 提升模型的**平移不变性**（Translation Invariance）；
* 抑制过拟合。

常见的池化方式有：

* **最大池化（Max Pooling）**：提取局部区域中的最大值；
* **平均池化（Average Pooling）**：提取局部区域的平均值。

---

### 3.2 数学公式

最大池化的过程相当于将输入划分为一系列不重叠或部分重叠的窗口区域，输出每个窗口内的最大值。

设输入张量为 \$X \in \mathbb{R}^{C \times H \times W}\$，窗口大小为 \$(k\_H, k\_W)\$，步长为 \$(s\_H, s\_W)\$，则输出为：

$$
Y_{n, c, i, j} = \max_{\substack{0 \leq u < k_H \\ 0 \leq v < k_W}} X_{n, c, i \cdot s_H + u, j \cdot s_W + v}
$$

其中 \$(i,j)\$ 是输出特征图上的空间坐标。

---

### 3.3 参数解释

1. `kernel_size`

* 池化窗口的大小，如 (2,2)、(3,3)。

2. `stride`

* 控制窗口移动的步长，默认为与 `kernel_size` 相同。

3. `padding`

* 边缘填充，控制输入尺寸是否补 0。

---

### 3.4 实现与优化细节

我们采用 `im2col` 技术对输入张量展开为一系列滑动窗口，再对每个窗口取最大值。

相比卷积中的 `im2col`：

* **卷积层的展开结果维度是 `[bs, oh*ow, ic*kh*kw]`**，为了矩阵乘法；
* **而池化层展开为 `[bs, c, oh*ow, kh*kw]`**，我们保留了每个通道独立，直接对每个窗口区域做 `max` 操作，计算更高效，逻辑更清晰。

代码中使用 `cols.max(axis=3)` 快速提取每个滑动窗口的最大值，避免了循环遍历。

---

### 3.5 应用场景

* 通常在卷积层后使用，用于特征图尺寸的压缩；
* 可作为结构中的“信息摘要器”；
* 对细节要求较低的任务如分类中常用 max pooling。

---

