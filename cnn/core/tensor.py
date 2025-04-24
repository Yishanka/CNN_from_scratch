import numpy as np

class Tensor:
    '''
    Tensor 类，用于表示多维数组，并支持自动求导。
    '''
    def __init__(self, data, requires_grad:bool=False, _children=(), _op:str=''):
        '''
        Parameters:
            data: 数据，转换为 numpy 数组
            requires_grad: 是否需要计算梯度，默认 False
            _children: 子节点，用于构建计算图，默认空元组
            _op: 操作符，用于标识该 Tensor 是如何生成的，默认空字符串
        '''
        self._data = np.array(data) # 将数据转换为 numpy 数组
        self._grad = np.zeros_like(self._data) if requires_grad else None  # 梯度初始化为零
        self._children = set(_children)  # 子节点集合
        self._op = _op  # 操作符
        self._backward = lambda: None  # 反向传播的梯度计算函数，默认为空
        
        self.shape = self._data.shape
        self.size = self._data.size
        self.requires_grad = requires_grad  # 是否需要计算梯度

    @property
    def grad(self):
        return Tensor(self._grad)
    
    @property
    def T(self):  
        '''对 Tensor，shape: m*n 进行转置操作，返回 Tensor，shape: n*m。'''
        out = Tensor(self._data.T, requires_grad=self.requires_grad, _children=(self,), _op='T')
        def _backward():
            if self.requires_grad:
                # 广播安全
                self._grad += out._grad.T
        out._backward = _backward
        return out
    
    def transpose(self, axes:tuple):
        '''
        Transpose tensor by permuting dimensions.
        '''
        out_data = self._data.transpose(axes)
        out = Tensor(out_data, requires_grad=self.requires_grad, _children=(self,), _op='transpose')

        def _backward():
            if self.requires_grad:
                # 逆转 axes：原 axes[i] = j → inverse_axes[j] = i
                inverse_axes = [0] * len(axes)
                for i, a in enumerate(axes):
                    inverse_axes[a] = i
                inverse_axes = tuple(inverse_axes)
                self._grad += out._grad.transpose(inverse_axes)

        out._backward = _backward
        return out

    def __repr__(self):
        # 格式化输出
        data_str = f"{self._data:.3f}" if np.isscalar(self._data) else np.array2string(self._data, precision=3, separator=', ')
        grad_str = f"{self._grad:.3f}" if np.isscalar(self._grad) else np.array2string(self._grad, precision=3, separator=', ') if self.requires_grad else None
        return f"data:\n{data_str}\ngrad:\n{grad_str}\n"
    
    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            idx = tuple(i._data if isinstance(i, Tensor) else i for i in idx)
        elif isinstance(idx, Tensor):
            idx = idx._data  # 单个 Tensor 索引
        out = Tensor(self._data[idx], requires_grad=self.requires_grad, _children=(self,), _op='getitem')

        def _backward():
            if self.requires_grad:
                grad = np.zeros_like(self._data)
                np.add.at(grad, idx, out._grad)  # 这行是 NumPy 的广播安全反向写法
                self._grad += grad
        out._backward = _backward

        return out
    
    def __add__(self, other):
        '''重载加法运算符，支持两个 Tensor，shape: m*n 相加，返回 Tensor，shape: m*n。'''
        other = other if isinstance(other, Tensor) else Tensor(other)
        if other.shape != self.shape:
            other = other.broadcast_to(self.shape)
        out = Tensor(
            self._data + other._data, 
            requires_grad=self.requires_grad or other.requires_grad, 
            _children=(self, other), 
            _op='+'
        )

        def _backward():
            # ∂f(x+y)/∂x=∂f(x+y)/∂(x+y)*∂(x+y)/∂x=∂f(x+y)/∂(x+y)
            if self.requires_grad:
                self._grad += out._grad
            if other.requires_grad:
                other._grad += out._grad
        out._backward = _backward

        return out

    def __radd__(self, other):
        '''重载加法运算符，支持两个 Tensor，shape: m*n 相加，返回 Tensor，shape: m*n。'''
        return self + other

    def __neg__(self):
        out = Tensor(
            -self._data, 
            requires_grad=self.requires_grad, 
            _children=(self,), 
            _op='neg'
        )

        def _backward():
            # ∂f(-x)/∂x=∂f(-x)/∂(-x)*∂(-x)/∂x=-∂f(x+y)/∂(x+y)
            if self.requires_grad:
                self._grad -= out._grad
        out._backward = _backward

        return out
    
    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self + (-other)
    
    def __rsub__(self, other):
        return (-self) + other
    
    def __mul__(self, other):
        '''重载乘法运算符，支持两个 Tensor，shape: m*n 逐元素相乘，返回 Tensor，shape: m*n。'''
        other = other if isinstance(other, Tensor) else Tensor(other)
        if other.shape != self.shape:
            other = other.broadcast_to(self.shape)
        out = Tensor(
            self._data * other._data, 
            requires_grad=self.requires_grad or other.requires_grad, 
            _children=(self, other), 
            _op='*'
        )
        def _backward():
            # ∂f(xy)/∂x=∂f(xy)/∂(xy)*∂(xy)/∂x=∂f(xy)/∂(xy)*y
            if self.requires_grad:
                self._grad += other._data * out._grad
            if other.requires_grad:
                other._grad += self._data * out._grad
        out._backward = _backward 
        return out
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __truediv__(self, other):
        '''重载除法运算符，支持两个 Tensor，shape: m*n 逐元素相除，返回 Tensor, shape: m*n。'''
        other = other if isinstance(other, Tensor) else Tensor(other)
        if other.shape != self.shape:
            other = other.broadcast_to(self.shape)
        out = Tensor(
            self._data / other._data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
            _op='/'
        )

        def _backward():
            if self.requires_grad:
                self._grad += (1 / other._data) * out._grad
            if other.requires_grad:
                other._grad += (-self._data / (other._data ** 2)) * out._grad
        out._backward = _backward
        return out
    
    def __rtruediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return other / self  # 反转调用
    
    def __pow__(self, power):
        power = power if isinstance(power, Tensor) else Tensor(power)
        if power.shape != self.shape:
            power = power.broadcast_to(self.shape)
        out = Tensor(self._data ** power._data, requires_grad=self.requires_grad, _children=(self, power), _op='**')

        def _backward():
            if self.requires_grad:
                self._grad += (power._data * self._data ** (power._data - 1)) * out._grad
            if power.requires_grad:
                self_power_log = np.log(self._data + 1e-10)  # 防止 log(0)
                power._grad += (self._data ** power._data) * self_power_log * out._grad
        out._backward = _backward

        return out
    
    def __matmul__(self, other):
        '''重载矩阵乘法运算符，支持 Tensor，shape: m*n 和 Tensor，shape: n*p 进行矩阵乘法，返回 Tensor，shape: m*p。'''
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self._data @ other._data, requires_grad=self.requires_grad or other.requires_grad, _children=(self, other), _op='matmul')
        def _backward():
            # ∂f(X/Y)/∂X=∂f(XY)/∂(XY)@∂(XY)/∂X=∂f(XY)/∂(XY)@Y.T
            if self.requires_grad:
                self._grad += out._grad @ other._data.T
            if other.requires_grad:
                other._grad += self._data.T @ out._grad
        out._backward = _backward
        return out
    
    def broadcast_to(self, shape):
        '''
        自动广播到目标形状，并在反向传播时正确累加梯度。
        '''
        out_data = np.broadcast_to(self._data, shape)
        out = Tensor(out_data, requires_grad=self.requires_grad, _children=(self,), _op='broadcast')

        def _backward():
            if self.requires_grad:
                grad = out._grad

                # === 补齐 self.shape 到目标 shape 的长度 ===
                self_shape = self._data.shape
                num_missing_dims = len(shape) - len(self_shape)
                padded_shape = (1,) * num_missing_dims + self_shape  # e.g. (1, 3, 1, 5)

                grad_axes = [
                    i for i, (s, t) in enumerate(zip(padded_shape, shape))
                    if s == 1 and t != 1
                ]

                if grad_axes:
                    grad = grad.sum(axis=tuple(grad_axes), keepdims=True)

                # 去掉多余的维度，使其回到 self._data 的 shape
                grad = grad.reshape(self_shape)

                self._grad += grad

        out._backward = _backward
        return out
    
    def sum(self, axis=None, keepdims=False):
        '''对 Tensor 的所有元素求和，返回标量 Tensor。'''
        out = Tensor(self._data.sum(axis=axis, keepdims=keepdims), requires_grad=self.requires_grad, _children=(self,), _op='sum')  # 创建新的 Tensor，表示求和结果
        
        def _backward():
            # ∂f(Σx)/∂x=∂f(Σx)/∂(Σx)*∂(Σx)/∂x=∂f(Σx)/∂(Σx)
            if self.requires_grad:
                grad = out._grad
                if not keepdims and axis:
                    grad = np.expand_dims(grad, axis)
                self._grad += np.ones_like(self._data) * grad
        out._backward = _backward  # 将反向传播函数绑定到输出 Tensor
        
        return out  # 返回求和结果
    
    def maximum(self, other):
        '''求出两个 Tensor 之间的最大值'''
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(np.maximum(self._data, other._data), requires_grad=self.requires_grad or other.requires_grad, _children=(self, other), _op='maximum')

        def _backward():
            if self.requires_grad:
                self._grad += (self._data >= other._data) * out._grad
            if other.requires_grad:
                other._grad += (self._data < other._data) * out._grad
        out._backward = _backward

        return out

    def max(self, axis=None, keepdims=False):
        '''求出 Tensor 按某一维度的最大值'''
        out = Tensor(self._data.max(axis=axis, keepdims=keepdims), requires_grad=self.requires_grad, _children = (self,), _op = 'max')

        def _backward():
            if self.requires_grad:
                max_vals = np.max(self._data, axis=axis, keepdims=True)
                mask = self._data == max_vals
                grad = mask * out._grad
                self._grad += grad
        out._backward = _backward

        return out
    
    def exp(self):
        out = Tensor(np.exp(self._data), requires_grad=self.requires_grad, _children=(self,), _op='exp')

        def _backward():
            if self.requires_grad:
                self._grad += out._data * out._grad
        out._backward = _backward

        return out
    
    def log(self):
        out = Tensor(np.log(self._data + 1e-10), requires_grad=self.requires_grad, _children=(self,), _op='log')  # 防止 log(0)

        def _backward():
            if self.requires_grad:
                self._grad += (1 / (self._data + 1e-10)) * out._grad
        out._backward = _backward

        return out
    
    def abs(self):
        out = Tensor(np.abs(self._data), requires_grad=self.requires_grad, _children=(self,), _op="abs")
        
        def _backward():
            if self.requires_grad:
                self._grad += np.sign(self._data) * out._grad
        out._backward = _backward
        
        return out

    def reshape(self, shape):
        '''
        返回新的 Tensor，其数据是 reshape 之后的（重排原数据），梯度反向传播会 reshape 回原来的形状。
        '''
        out_data = self._data.reshape(shape)
        out = Tensor(out_data, requires_grad=self.requires_grad, _children=(self,), _op='reshape')

        def _backward():
            if self.requires_grad:
                # 把梯度 reshape 回原来的形状
                self._grad += out._grad.reshape(self.shape)

        out._backward = _backward
        return out
    
    def pad(self, pad_width: tuple[tuple[int, int], ...]):
        '''
        通用 zero-padding，封装 np.pad 接口。
        
        Parameters:
            pad_width: 与 np.pad 一致，例如：
                ((0, 0), (0, 0), (1, 2), (3, 3)) 表示：
                - 第三维前 pad 1，后 pad 2
                - 第四维前 pad 3，后 pad 3
        '''
        # todo: 没用tensor的索引，可选择改进
        # 安全性校验
        assert all(isinstance(p, tuple) and len(p) == 2 for p in pad_width), "每个维度必须是 (before, after)"
        
        # 自动补足维度
        ndim = len(self.shape)
        pad_width = list(pad_width)
        while len(pad_width) < ndim:
            pad_width.append((0, 0))
        if len(pad_width) > ndim:
            pad_width = pad_width[:ndim]
        pad_width = tuple(pad_width)

        # 使用 np.pad 进行前向计算
        out_data = np.pad(self._data, pad_width, mode='constant')

        out = Tensor(out_data, requires_grad=self.requires_grad, _children=(self,), _op='pad')

        # 构造反向传播：裁剪掉 padding 区域
        slices = tuple(
            slice(p[0], p[0] + s) for p, s in zip(pad_width, self.shape)
        )

        def _backward():
            if self.requires_grad:
                grad = out._grad[slices]
                self._grad += grad

        out._backward = _backward
        return out
    
    def stack(tensors, axis=0):
        '''将一组 Tensor 沿指定维度拼接成一个新 Tensor。'''
        assert all(isinstance(t, Tensor) for t in tensors), "所有元素必须是 Tensor"

        data = np.stack([t._data for t in tensors], axis=axis)
        requires_grad = any(t.requires_grad for t in tensors)
        out = Tensor(data, requires_grad=requires_grad, _children=tuple(tensor for tensor in tensors), _op='pad')

        def _backward():
            if requires_grad:
                grads = np.split(out._grad, len(tensors), axis=axis)
                for t, g in zip(tensors, grads):
                    t._grad += g.squeeze(axis=axis)

        out._backward = _backward
        return out

    def zero_grad(self):
        '''将梯度清零。'''
        if self.requires_grad:
            self._grad = np.zeros_like(self._data)

    def backward(self, retain_graph=False):
        ''' 反向传播 '''
        # 初始化 loss 的导数为 1
        self._grad =np.ones_like(self._data)

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
            if not retain_graph:
                node._children.clear()
                node._backward = lambda: None

    def zeros(shape, requires_grad=False):
        '''
        创建一个给定 shape 的零张量。
        '''
        return Tensor(np.zeros(shape), requires_grad=requires_grad)

    def zeros_like(data):
        '''返回一个全零 numpy 数组，shape 保持一致，不参与反向传播'''
        if isinstance(data, Tensor):
            data = data._data
        return Tensor(np.zeros_like(data))
    


def sqrt(data):
    data = data if isinstance(data, Tensor) else Tensor(data)
    return data ** 0.5


def sum(data, axis=None, keepdims=False):
    '''对 Tensor 的所有元素求和，返回标量 Tensor。'''
    data = data if isinstance(data, Tensor) else Tensor(data)
    return data.sum(axis=axis, keepdims=keepdims)


def max(data, other):
    '''求出两个 Tensor 之间的最大值'''
    data = data if isinstance(data, Tensor) else Tensor(data)
    return data.max(other)

  
def exp(data):
    data = data if isinstance(data, Tensor) else Tensor(data)
    return data.exp()


def log(data):
    data = data if isinstance(data, Tensor) else Tensor(data)
    return data.log()

def abs(data):
    data = data if isinstance(data, Tensor) else Tensor(data)
    return data.abs()


if __name__ == '__main__':
    # tensor_a = Tensor([[1,2,2]], requires_grad=True)
    # tensor_b = tensor_a.repeat(axis=1,repeats=10)
    # print(tensor_b)
    a = Tensor([1,2,3,4],requires_grad=True)
    print(a.shape)
    a = a.reshape((4, -1))
    print(a)
