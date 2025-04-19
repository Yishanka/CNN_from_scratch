# loss 模块文档
## 均方误差损失函数（Mean Square Error Loss)

## 交叉熵损失函数（Cross Entropy Loss）
### 概念简述

**交叉熵损失函数**（Cross Entropy Loss）是用于分类问题的经典损失函数，尤其适用于多类别分类任务（multi-class classification）。

它衡量了模型输出的概率分布（通常通过 softmax 得到）与真实分布之间的“距离”，越接近则损失越小。

---

### 数学定义

设：

- $ \hat{y}_i \in \mathbb{R}^C $：第 $ i $ 个样本的模型预测概率向量（softmax 输出）
- $ y_i \in \{0, 1, ..., C-1\} $：第 $ i $ 个样本的真实类别（索引）
- $ N $：样本个数，$ C $：类别个数

则交叉熵损失定义为：

$$
\mathcal{L} = -\frac{1}{N} \sum_{i=1}^N \log(\hat{y}_{i, y_i})
$$

即，对每个样本选出其正确类别对应的预测概率，取负对数，求平均。

---

### 代码重点部分解释
- `range(batch_size), true`：用于高级索引（fancy indexing），从每行中选出对应标签类别的概率。

---
### 注意事项
- 模型最后一层输出经过 softmax 归一化，输入 `pred` 应当是 softmax 概率，不是 logits。
- 标签必须是一维整数索引；如有 one-hot 需要先 `argmax`，开发者后续可实现这部分内容。

---

### 🧭 后续可拓展方向

- 加入 `ignore_index` 参数支持“标签掩码”。
- 支持 `label smoothing` 平滑标签。
- 支持 `one-hot` 独热编码
- 实现 `F.cross_entropy` 风格的 logits + 内部 softmax 自动化。

---