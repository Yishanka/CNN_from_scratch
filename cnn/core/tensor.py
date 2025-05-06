import numpy as np
# todo: 需要为每个方法加上一个判断其结果是否需要参与到反向传播的过程中，构建计算图时忽略这个 Tensor
# todo: 将前向传播中调用的 broadcast 修改为反向传播中的 unbroadcast，减少计算图深度
class Tensor:
    '''
    Tensor 类，用于表示多维数组，并支持自动求导。
    '''
    def __init__(self, data, requires_grad:bool=False, children=(), op:str='', graph:bool=True):
        '''
        Parameters:
            data: 数据，转换为 numpy 数组
            requires_grad: 是否需要计算梯度，默认 False
            _children: 子节点，用于构建计算图，默认空元组
            _op: 操作符，用于标识该 Tensor 是如何生成的，默认空字符串
        '''
        self.data = np.array(data, dtype=np.float64) # 将数据转换为 numpy 数组
        self._grad = np.zeros_like(self.data, dtype=np.float64) if requires_grad else None  # 梯度初始化为零
        self._children = set(children)  # 子节点集合
        self._op = op  # 操作符
        self._backward = lambda: None  # 反向传播的梯度计算函数，默认为空
        self._graph = graph
        
        self.shape = self.data.shape
        self.size = self.data.size
        self.requires_grad = requires_grad  # 是否需要计算梯度

    @property
    def grad(self):
        return Tensor(self._grad)
    
    @property
    def T(self):  
        '''对 Tensor，shape: m*n 进行转置操作，返回 Tensor，shape: n*m。'''
        out = Tensor(self.data.T, requires_grad=self.requires_grad, children=(self,), op='T')
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
        out_data = self.data.transpose(axes)
        out = Tensor(out_data, requires_grad=self.requires_grad, children=(self,), op='transpose')

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
        data_str = f"{self.data:.3f}" if np.isscalar(self.data) else np.array2string(self.data, precision=3, separator=', ')
        grad_str = f"{self._grad:.3f}" if np.isscalar(self._grad) else np.array2string(self._grad, precision=3, separator=', ') if self.requires_grad else None
        return f"data:\n{data_str}\ngrad:\n{grad_str}\n"
    
    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            idx = tuple(i.data if isinstance(i, Tensor) else i for i in idx)
        elif isinstance(idx, Tensor):
            idx = idx.data  # 单个 Tensor 索引
        out = Tensor(self.data[idx], requires_grad=self.requires_grad, children=(self,), op='getitem')

        def _backward():
            if self.requires_grad:
                grad = np.zeros_like(self.data)
                np.add.at(grad, idx, out._grad)  # 这行是 NumPy 的广播安全反向写法
                self._grad += grad
        out._backward = _backward

        return out
    
    def __add__(self, other):
        '''重载加法运算符，支持两个 Tensor，shape: m*n 相加，返回 Tensor，shape: m*n。'''
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, requires_grad=self.requires_grad or other.requires_grad, children=(self, other), op='+')

        def _backward():
            if self.requires_grad:
                self._grad += unbroadcast(out._grad, self.shape)
            if other.requires_grad:
                other._grad += unbroadcast(out._grad, other.shape)

        out._backward = _backward
        return out

    def __radd__(self, other):
        '''重载加法运算符，支持两个 Tensor，shape: m*n 相加，返回 Tensor，shape: m*n。'''
        return self + other

    def __neg__(self):
        out = Tensor(
            -self.data, 
            requires_grad=self.requires_grad, 
            children=(self,), 
            op='neg'
        )

        def _backward():
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
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            self.data * other.data, 
            requires_grad=self.requires_grad or other.requires_grad, 
            children=(self, other), 
            op='*'
        )
        def _backward():
            if self.requires_grad:
                self._grad += unbroadcast(other.data * out._grad, self.shape)
            if other.requires_grad:
                other._grad += unbroadcast(self.data * out._grad, other.shape)
        out._backward = _backward 
        return out
        
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            self.data / other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            children=(self, other),
            op='/'
        )

        def _backward():
            if self.requires_grad:
                self._grad += unbroadcast((1 / other.data) * out._grad, self.shape)
            if other.requires_grad:
                other._grad += unbroadcast((-self.data / (other.data ** 2)) * out._grad, other.shape)
        out._backward = _backward
        return out
    
    def __rtruediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return other / self  # 反转调用
    
    def __pow__(self, power):
        power = power if isinstance(power, Tensor) else Tensor(power)
        out = Tensor(self.data ** power.data, requires_grad=self.requires_grad or power.requires_grad, children=(self, power), op='**')

        def _backward():
            if self.requires_grad:
                self._grad += unbroadcast((power.data * self.data ** (power.data - 1)) * out._grad, self.shape)
            if power.requires_grad:
                safe_log = np.log(self.data + 1e-10)  # 防止 log(0)
                power._grad += unbroadcast((self.data ** power.data) * safe_log * out._grad, power.shape)
        out._backward = _backward

        return out

    def __matmul__(self, other):
        
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data, requires_grad=self.requires_grad or other.requires_grad, children=(self, other), op='matmul')
        
        def _backward():
            if self.requires_grad:
                grad = out._grad @ other.data.transpose((*range(other.data.ndim - 2), -1, -2))
                while len(grad.shape) > len(other.data.shape):
                    grad = grad.sum(axis=0)
                self._grad += grad
            if other.requires_grad:
                grad = self.data.transpose((*range(self.data.ndim - 2), -1, -2)) @ out._grad
                while len(grad.shape) > len(other.data.shape):
                    grad = grad.sum(axis=0)
                other._grad += grad
        out._backward = _backward
        
        return out
    
    # def broadcast_to(self, shape):
    #     '''
    #     自动广播到目标形状，并在反向传播时正确累加梯度。
    #     '''
    #     out_data = np.broadcast_to(self.data, shape)
    #     out = Tensor(out_data, requires_grad=self.requires_grad, children=(self,), op='broadcast')

    #     def _backward():
    #         if self.requires_grad:
    #             grad = out._grad

    #             self_shape = self.data.shape
    #             num_missing_dims = len(shape) - len(self_shape)
    #             padded_shape = (1,) * num_missing_dims + self_shape  # e.g. (1, 3, 1, 5)

    #             grad_axes = [
    #                 i for i, (s, t) in enumerate(zip(padded_shape, shape))
    #                 if s == 1 and t != 1
    #             ]

    #             if grad_axes:
    #                 grad = grad.sum(axis=tuple(grad_axes), keepdims=True)

    #             # 去掉多余的维度，使其回到 self.data 的 shape
    #             grad = grad.reshape(self_shape)

    #             self._grad += grad

    #     out._backward = _backward
    #     return out
    
    def sum(self, axis=None, keepdims=False):
        '''对 Tensor 的所有元素求和，返回标量 Tensor。'''
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims,dtype=np.float64), requires_grad=self.requires_grad, children=(self,), op='sum')  # 创建新的 Tensor，表示求和结果
        
        def _backward():
            # ∂f(Σx)/∂x=∂f(Σx)/∂(Σx)*∂(Σx)/∂x=∂f(Σx)/∂(Σx)
            if self.requires_grad:
                grad = out._grad
                if not keepdims and axis:
                    grad = np.expand_dims(grad, axis)
                self._grad += np.ones_like(self.data) * grad
        out._backward = _backward  # 将反向传播函数绑定到输出 Tensor
        
        return out  # 返回求和结果
    
    def maximum(self, other):
        '''求出两个 Tensor 之间的最大值'''
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(np.maximum(self.data, other.data,dtype=np.float64), requires_grad=self.requires_grad or other.requires_grad, children=(self, other), op='maximum')

        def _backward():
            if self.requires_grad:
                self._grad += (self.data >= other.data) * out._grad
            if other.requires_grad:
                other._grad += (self.data < other.data) * out._grad
        out._backward = _backward

        return out

    def max(self, axis=None, keepdims=False):
        '''求出 Tensor 按某一维度的最大值'''
        out = Tensor(self.data.max(axis=axis, keepdims=keepdims), requires_grad=self.requires_grad, children = (self,), op = 'max')

        def _backward():
            if self.requires_grad:
                grad = out._grad
                if not keepdims and axis is not None:
                    grad = np.expand_dims(grad, axis=axis)
                max_vals = self.data.max(axis=axis, keepdims=True)
                mask = self.data == max_vals
                count = np.sum(mask, axis=axis, keepdims=True, dtype=np.float64)  # 对最大值均分梯度
                self._grad += mask * grad / count
        out._backward = _backward

        return out
    
    def exp(self):
        out = Tensor(np.exp(self.data), requires_grad=self.requires_grad, children=(self,), op='exp')

        def _backward():
            if self.requires_grad:
                self._grad += out.data * out._grad
        out._backward = _backward

        return out
    
    def log(self):
        out = Tensor(np.log(self.data + 1e-10,dtype=np.float64), requires_grad=self.requires_grad, children=(self,), op='log')  # 防止 log(0)

        def _backward():
            if self.requires_grad:
                self._grad += (1 / (self.data + 1e-10)) * out._grad
        out._backward = _backward

        return out
    
    def abs(self):
        out = Tensor(np.abs(self.data, dtype=np.float64), requires_grad=self.requires_grad, children=(self,), op="abs")
        
        def _backward():
            if self.requires_grad:
                self._grad += np.sign(self.data) * out._grad
        out._backward = _backward
        
        return out

    def reshape(self, shape):
        '''
        返回新的 Tensor，其数据是 reshape 之后的（重排原数据），梯度反向传播会 reshape 回原来的形状。
        '''
        out_data = self.data.reshape(shape)
        out = Tensor(out_data, requires_grad=self.requires_grad, children=(self,), op='reshape')

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
        out_data = np.pad(self.data, pad_width, mode='constant')

        out = Tensor(out_data, requires_grad=self.requires_grad, children=(self,), op='pad')

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

        data = np.stack([t.data for t in tensors], axis=axis, dtype=np.float64)
        requires_grad = any(t.requires_grad for t in tensors)
        out = Tensor(data, requires_grad=requires_grad, children=tuple(tensor for tensor in tensors), op='pad')

        def _backward():
            if requires_grad:
                grads = np.split(out._grad, len(tensors), axis=axis)
                for t, g in zip(tensors, grads):
                    t._grad += g.squeeze(axis=axis)

        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        data = self.data.mean(axis=axis, keepdims=keepdims)
        out = Tensor(data, requires_grad=self.requires_grad, children=(self,), op='mean')

        def _backward():
            if self.requires_grad:
                grad = out._grad
                # 需要将 grad 广播回原始形状
                factor = np.prod(self.data.shape) / np.prod(data.shape)
                if not keepdims:
                    shape = list(self.data.shape)
                    if isinstance(axis, int):
                        shape[axis] = 1
                    elif isinstance(axis, tuple):
                        for ax in axis:
                            shape[ax] = 1
                    grad = grad.reshape(shape)
                self._grad += np.broadcast_to(grad, self.data.shape) / factor

        out._backward = _backward
        return out

    def var(self, axis=None, keepdims=False):
        mean = self.mean(axis=axis, keepdims=True)
        centered = self - mean
        squared = centered * centered
        return squared.mean(axis=axis, keepdims=keepdims)
    
###################################################
    
    def zero_grad(self):
        '''将梯度清零。'''
        if self.requires_grad:
            self._grad = np.zeros_like(self.data, dtype=np.float64)
        else:
            self._grad = None

    def backward(self):
        ''' 反向传播，默认不保留计算图 '''
        # 初始化 loss 的导数为 1
        self._grad =np.ones_like(self.data)

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

    def remove_graph(self):
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
            node._children=set()
            node._backward = lambda: None

    def zeros(shape, requires_grad=False):
        '''
        创建一个给定 shape 的零张量。
        '''
        return Tensor(np.zeros(shape,dtype=np.float64), requires_grad=requires_grad)

    def zeros_like(data):
        '''返回一个全零张量，shape 保持一致，不参与反向传播'''
        if isinstance(data, Tensor):
            data = data.data
        return Tensor(np.zeros_like(data,dtype=np.float64))
    
    def ones(shape, requires_grad=False):
        '''
        创建一个给定 shape 的零张量。
        '''
        return Tensor(np.ones(shape,dtype=np.float64), requires_grad=requires_grad)
    
    def argmax(self, axis=None, keepdims=False):
        '''不参与求导计算'''
        out = self.data.argmax(axis=axis, keepdims=keepdims)
        return out
    
    def to_int(self):
        '''原地修改数据类型到int, 不参与梯度下降'''
        self.data = np.int32(self.data)
        
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
    # a = Tensor([1,2,3,4],requires_grad=True)
    # print(a.shape)
    # a = a.reshape((4, -1))
    # print(a)
    a = np.array([[[1,1,1], [1,1,1]]])
    b = np.array([[1,1,1,1],[1,1,1,1],[1,1,1,1]])
    c = a@b
    print(c)