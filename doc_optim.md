## 1. 优化方法

自实现框架的计算速度相比 PyTorch 等现有框架很慢。
在避免增强硬件支持（如 gpu 使用，AVX512 矢量化计算等）情况下，我们在采用到如下的优化方法以尽量提高自实现框架的计算速度

#### 1.1. 广播机制与反向传播

- 有些计算中，维度不匹配时，需要进行广播操作。
- 基于 `numpy` 的隐式广播会导致反向传播时，“大梯度”无法放回“小梯度”，无法正确计算。
- 而实现 `Tensor` 类的 `broadcast` 函数并在前向传播时显示调用，`broadcast` 作为计算结点加入到计算图中
- 反向传播会大幅增加计算图长度。

**因此，反向传播中将梯度还原，可以正确且高效地传播梯度**

梯度还原 `unbroadcast` 实现如下：

```python
def unbroadcast(grad, shape):
    '''
    将广播后的梯度 grad 还原到原始形状 shape。
    '''
    if grad.shape == shape:
        return grad
    
    # 计算补齐后的形状
    num_missing_dims = grad.ndim - len(shape)
    padded_shape = (1,) * num_missing_dims + shape

    axes = [i for i, (g_dim, s_dim) in enumerate(zip(grad.shape, padded_shape)) if s_dim == 1]

    if axes:
        grad = grad.sum(axis=tuple(axes), keepdims=True)

    return grad.reshape(shape)
```

---

#### 1.2. 计算图剪枝：删除所有不需要求导的结点

在计算图中，剪枝是一种优化技术，用于删除所有不需要求导的结点。
    - 这是因为不需要求导的结点的子结点也不需要求导；
    - 因此整个子图都可以被剪枝，从而减少循环次数。

具体实现如下：
```python
def backward(self, remove_graph=True):
    ''' 反向传播 '''
    # 初始化 loss 的导数为 1
    self.grad = np.ones_like(self.data)
    # 拓扑排序
    topo: list[Tensor] = []
    visited = set()
    def build_topo(t: Tensor):
        if t not in visited and t.requires_grad:  # 不需要求梯度的 Tensor 不需要进入计算图
            visited.add(t)
            for child in t._children:
                build_topo(child)
            topo.append(t)
    build_topo(self)
    # 按拓扑顺序执行每个 tensor 的 _backward，开始反向传播
    for node in reversed(topo):
        node._backward()
        if remove_graph:
            node._children = tuple()
            node._backward = lambda: None
```

---

#### 1.3. 特殊方法：`as_strided`

`as_strided` 是一种高效的内存视图操作，常用于优化 `im2col` 等需要数据重排的操作。通过创建视图而非实际拷贝数据，可以显著减少内存开销，以及由循环重构视图及其连带的额外计算结点带来的开销。

具体实现如下：
```python
def _as_strided(self, shape, strides):
    '''创建视图'''
    out = Tensor(np.lib.stride_tricks.as_strided(self.data, shape=shape, strides=strides), requires_grad=self.requires_grad, children=(self,))
    if self.requires_grad:
        grad_view = np.lib.stride_tricks.as_strided(self.grad, shape=shape, strides=strides)
        def _backward():
            np.add.at(grad_view, ..., out.grad)
        out._backward = _backward
    return out
```

---

#### 1.4. 避免临时变量优化
若在反向传播时先创建一个临时变量来保存梯度，然后再将其分配到原始张量上，会引入额外的内存分配和拷贝操作，导致性能下降。

通过直接操作梯度，避免了临时变量的创建，从而显著提升了性能。

以 `__getitem__` 为例，具体实现如下：
```python
def __getitem__(self, idx: tuple):
    idx = idx if isinstance(idx, tuple) else tuple(idx)
    out = Tensor(self.data[idx], requires_grad=self.requires_grad, children=(self,))
    if self.requires_grad:
        def _backward():
            np.add.at(self.grad, idx, out.grad)
        out._backward = _backward
    return out
```

---

#### 1.5. 预先计算与判断
在前向传播的过程中，会出现许多不需要求梯度的结点；虽然它们已经不被加入到反向传播的序列中，但是前向传播时函数闭包的创建也有一定开销
因此将是否需要反向传播的判断放在 `_backward` 之前，而不是在反向传播的过程中判断

同时，辅助计算的临时变量也可以放在 `_backward` 之前，避免同一结点多次累加梯度时反复计算这些临时变量

具体实现如下(以 `pad` 为例)：
```python
    def pad(self, pad_width: tuple[tuple[int, int], ...]):
        # 安全性校验
        assert all(isinstance(p, tuple) and len(p) == 2 for p in pad_width), '每个维度必须是 (before, after)'
        
        # 维度对齐优化
        ndim = self.data.ndim
        pad_width = tuple(pad_width[i] if i < len(pad_width) else (0, 0) for i in range(ndim))

        out = Tensor(np.pad(self.data, pad_width, mode='constant'), requires_grad=self.requires_grad, children=(self,))

        # 优先判断是否需要创建反向传播函数
        if self.requires_grad:
            # 优先计算临时变量
            slices = tuple(slice(p[0], p[0] + s) for p, s in zip(pad_width, self.shape))
            def _backward():
                np.add(self.grad, out.grad[slices], out=self.grad)
            out._backward = _backward
        return out
```

---

#### 1.6. 使用 `einsum` 替代手动转置与降维优化矩阵乘法反向传播

在实现矩阵乘法的反向传播时，常规做法是先转置矩阵，再使用 `matmul` 执行反向计算，并处理因广播引起的维度差异（例如通过手动 `reshape`、`sum` 降维等）。这一过程不仅繁琐，而且会带来额外的内存开销和可读性问题。

为提升效率与简洁性，可使用 NumPy 的 `einsum`（爱因斯坦求和约定）直接表达张量乘积与转置组合操作，从而在一次调用中完成转置与乘法，避免显式中间变量，同时增强代码的通用性。

优化关键包括：

* **提前判断是否需要梯度传播**，在创建 `_backward` 函数之前判断 `requires_grad`，避免无用计算；
* **提前计算转置轴顺序**，避免在每次反向传播时重复调用 `transpose`;
* **使用 `einsum` 简化转置与乘积逻辑**，避免不必要的中间变量；
* **自动广播降维处理**，只在维度不一致时执行 `sum`。

优化后的 `__matmul__` 实现如下：

```python
def __matmul__(self, other):
    other = other if isinstance(other, Tensor) else Tensor(other)
    out_data = np.matmul(self.data, other.data)
    out = Tensor(out_data, requires_grad=self.requires_grad or other.requires_grad, children=(self, other))

    if out.requires_grad:
        # 提前计算转置轴顺序
        self_axes = tuple(range(self.data.ndim - 2)) + (-1, -2)
        other_axes = tuple(range(other.data.ndim - 2)) + (-1, -2)

        def _backward():
            if self.requires_grad:
                grad = np.einsum('...ij,...jk->...ik', out.grad, other.data.transpose(other_axes))
                extra_dims = grad.ndim - self.data.ndim
                if extra_dims > 0:
                    grad = grad.sum(axis=tuple(range(extra_dims)))
                np.add(self.grad, grad, out=self.grad)
            
            if other.requires_grad:
                grad = np.einsum('...ij,...jk->...ik', self.data.transpose(self_axes), out.grad)
                extra_dims = grad.ndim - other.data.ndim
                if extra_dims > 0:
                    grad = grad.sum(axis=tuple(range(extra_dims)))
                np.add(other.grad, grad, out=other.grad)

        out._backward = _backward

    return out
```

该优化方式不仅提升了运算效率与代码可读性，同时避免了手动处理张量形状与转置带来的出错风险。

---

#### 1.7.设置模型 `train`/`eval` 模式
在训练过程中，由于有反向传播操作，计算图可以在传播的过程中删除。

但是在验证/测试的过程中，没有反向传播操作，构建的计算图需要手动删除。

以 `adam` 为例，由于 python 的引用机制，其中创建的临时张量会由于长期存在的动量参数被永久引用，导致无法释放内存，造成内存泄漏

而手动删除会带来不必要的建图和删图的操作

模型中，只有参数张量是在定义时就需要保证允许求导的；通过设置模型 `train`/`eval` 模式，可以显示改变参数是否需要求导的属性，保证其在验证/测试时不需要求导

具体实现如下 (由 model 传到 layer，再由 layer 直接控制参数):
```python
def train(self):
    '''设置为训练模式'''
    self.training = True
    for param in self._params:
        param.requires_grad = True
    
def eval(self):
    '''设置为评估模式'''
    self.training = False
    for param in self._params:
        param.requires_grad = False
```

#### 其他方法
1. `np.add` 等基于 `numpy` 的显示原地计算，尽量避免临时内存创建
    效果并没有比 += 好很多

2. 原地操作的使用，减少结点创建，理论上可以大幅加快计算速度
    
    **实现时有许多问题待解决**
        - 单个张量对象重复参与原地计算和其他计算时，梯度可能无法得到正确累加
        - 改变形状的原地操作，在没有反向传播时（测试），模型参数即其他永久变量的形状被永久修改，计算错误
            - 目前只有 `transpose` 操作使用了原地操作，因为可以保证使用 `transpose` 的都是创建出的临时变量
    
    对单一不改变形状的计算流，原地操作可能带来较大的性能提升，如
    ```python
    def func(x):
        x += a
        x -= b
        x *= c

        return x
    ```
    未避免 bug，暂时不使用这种操作

3. 使用低精度的 `np.float32` 可以极大的加快速度，同时低精度数据似乎带来了一定的微小随机噪声，一定程度上增加了模型的泛化能力 