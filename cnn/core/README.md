# 核心组件：Tensor 支持自动求导的张量类文档

本模块实现了一个简洁的自动微分系统，核心目标是在不依赖外部框架（如 PyTorch、TensorFlow）的情况下，实现在 CNN 中所需要使用的基本张量操作与梯度传播机制，是从零实现 CNN 的基础。其设计核心是**基于计算图的反向传播算法**，通过操作的组合构建一棵有向无环图（DAG），并通过链式法则从输出节点向输入节点传递梯度。

---

## 核心结构：Tensor 类

### 核心属性说明

* `data`：存储张量的数值（NumPy 数组）。
* `requires_grad`：布尔值，指示是否需要对该张量求导。
* `grad`：该张量的梯度，初始为 `None`，反向传播过程中自动填充。
* `_children`：一个包含其依赖张量的的元组。
* `_backward`：一个函数指针，记录当前操作对应的反向传播函数。
* `_op`：字符串，记录当前 Tensor 是由哪种操作构建的，用于调试。。

---

## 计算图构建机制

每一个对 Tensor 的操作（如加法、乘法、矩阵乘法等）都会返回一个新的 Tensor。若操作的任一输入为 `requires_grad=True`，则新的 Tensor 会记录：

* 其依赖的子节点（即参与该操作的张量）
* 对应的反向传播函数（`_backward`）

这样就构成了一张**计算图**，从最终的 loss（输出 Tensor）出发，向输入逐步展开依赖。

---

## 分解单元计算与链式求导法则

反向传播依赖链式法则，将复杂运算分解为一系列基础操作。每个操作单元负责：

1. **定义前向计算**
2. **定义局部梯度计算**

### 示例：加法操作的求导

```python
    def __add__(self, other):
        '''重载加法运算符，支持两个 Tensor，shape: m*n 相加，返回 Tensor，shape: m*n。'''
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, requires_grad=self.requires_grad or other.requires_grad, children=(self, other), op='+')

        def _backward():
            if self.requires_grad:
                self.grad += unbroadcast(out.grad, self.shape)
            if other.requires_grad:
                other.grad += unbroadcast(out.grad, other.shape)

        out._backward = _backward
        return out


* 每个输入张量根据自身是否需要梯度，选择是否传递梯度。
```


## 反向传播执行过程：`backward()` 方法

核心流程如下：

1. **初始化输出节点的梯度为全 1**
2. **使用深度优先遍历构建计算图的拓扑序**
3. **逆拓扑顺序逐步调用 `_backward`，将梯度回传给每个子节点**
4. **若某节点已经存在梯度，则累加**

```python

    def backward(self):
        ''' 反向传播，默认不保留计算图 '''
        # 初始化 loss 的导数为 1
        self.grad =np.ones_like(self.data)

        # 拓扑排序
        topo: list[Tensor] = []
        visited = set()
        def build_topo(t: Tensor):
            if t not in visited:
                visited.add(t)
                for child in t._children:
                    build_topo(child)
                topo.append(t)
        build_topo(self)

        # 按拓扑顺序执行每个 tensor 的 _backward，开始反向传播
        for node in reversed(topo):
            node._backward()
            node._children=set()
            node._backward = lambda: None
```

* 每个 `Tensor` 的 `_backward` 负责计算本节点对子节点的梯度，并根据需要合并到子节点上。
* 多次使用同一张量时，梯度会自动累加。

---

## 广播机制与反向传播

广播操作在正向传播中可能改变维度，在前向传播时就需要时需将梯度还原（`unbroadcast`）：

```python
def unbroadcast(grad, shape):
    '''
    将广播后的梯度 grad 还原到原始形状 shape。
    '''
    # 计算补齐后的形状
    num_missing_dims = grad.ndim - len(shape)
    padded_shape = (1,) * num_missing_dims + shape

    axes = [i for i, (g_dim, s_dim) in enumerate(zip(grad.shape, padded_shape)) if s_dim == 1]

    if axes:
        grad = grad.sum(axis=tuple(axes), keepdims=True)

    return grad.reshape(shape)
```

* 将多余的维度（broadcast 产生）通过 `sum` 消除，恢复原始形状。
* 确保链式法则中的梯度维度一致。

---

## 总结

本自动求导框架核心思想是：

* 每个张量操作都会将其依赖关系与求导函数记录为一个计算图节点；
* 每个反向传播函数负责实现该操作对应的求导规则；
* 通过逆拓扑顺序遍历计算图，实现全局梯度回传；
* 广播机制在反向传播时通过形状还原补偿。

---

## 附录：优化方法

