import numpy as np
# todo: 需要为每个方法加上一个判断其结果是否需要参与到反向传播的过程中，构建计算图时忽略这个 Tensor
# todo: 将前向传播中调用的 broadcast 修改为反向传播中的 unbroadcast，减少计算图深度
class Tensor:
    '''
    Tensor 类，用于表示多维数组，并支持自动求导。
    '''
    def __init__(self, data, requires_grad:bool=False, children=(), op:str=''):
        '''
        Parameters:
            data: 数据，转换为 numpy 数组
            requires_grad: 是否需要计算梯度，默认 False
            _children: 子节点，用于构建计算图，默认空元组
            _op: 操作符，用于标识该 Tensor 是如何生成的，默认空字符串
        '''
        self.requires_grad = requires_grad  # 是否需要计算梯度
        self.data = np.ascontiguousarray(data, dtype=np.float64) # 将数据转换为 numpy 数组
        self.grad = np.zeros_like(self.data, dtype=np.float64) if self.requires_grad else None  # 梯度初始化为零
        self._children = set(children) if self.requires_grad else set() # 子节点集合
        self._op = op  # 操作符
        self._backward = lambda: None  # 反向传播的梯度计算函数，默认为空

        self.shape = self.data.shape
        self.size = self.data.size
      
    def __repr__(self):
        # 格式化输出
        data_str = f"{self.data:.4f}" if np.isscalar(self.data) else np.array2string(self.data, precision=3, separator=', ')
        grad_str = f"{self.grad:.4f}" if np.isscalar(self.grad) else np.array2string(self.grad, precision=3, separator=', ') if self.requires_grad else None
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
                np.add.at(grad, idx, out.grad)  # 这行是 NumPy 的广播安全反向写法
                self.grad += grad
        out._backward = _backward

        return out
    
    @property
    def T(self):  
        '''对 Tensor，shape: m*n 进行转置操作，返回 Tensor，shape: n*m。'''
        out = Tensor(self.data.T, requires_grad=self.requires_grad, children=(self,), op='T')
        
        if self.requires_grad:
            def _backward():
                # 广播安全
                self.grad += out.grad.T
            out._backward = _backward
        
        return out
    
    def transpose(self, axes: tuple):
        '''
        通过置换维度来转置张量。
        '''
        out = Tensor(self.data.transpose(axes), requires_grad=self.requires_grad, children=(self,), op='transpose')

        if self.requires_grad:
            inverse_axes = tuple(np.argsort(axes))  # More efficient than manual loop

            def _backward():
                self.grad += out.grad.transpose(inverse_axes)

            out._backward = _backward
        return out
      
    def __add__(self, other):
        '''重载加法运算符，支持两个 Tensor，shape: m*n 相加，返回 Tensor，shape: m*n。'''
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, requires_grad=self.requires_grad or other.requires_grad, children=(self, other), op='+')

        if out.requires_grad:
            # 静态定义反向传播函数（避免重复创建闭包）
            def _backward():
                if self.requires_grad:
                    self.grad += unbroadcast(out.grad, self.shape)
                if other.requires_grad:
                    other.grad += unbroadcast(out.grad, other.shape)
            out._backward = _backward
            
        return out

    def __radd__(self, other):
        '''重载加法运算符，支持两个 Tensor，shape: m*n 相加，返回 Tensor，shape: m*n。'''
        return self + other

    def __neg__(self):
        out = Tensor(-self.data, requires_grad=self.requires_grad, children=(self,), op='neg')

        if self.requires_grad:
            def _backward():
                    self.grad -= out.grad
            out._backward = _backward

        return out
    
    def __sub__(self, other):
        '''重载减法运算符，支持两个 Tensor，shape: m*n 相减，返回 Tensor，shape: m*n。'''
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data - other.data, requires_grad=self.requires_grad or other.requires_grad, children=(self, other), op='-')

        if out.requires_grad:
            # 静态定义反向传播函数（避免重复创建闭包）
            def _backward():
                if self.requires_grad:
                    self.grad += unbroadcast(out.grad, self.shape)
                if other.requires_grad:
                    other.grad -= unbroadcast(out.grad, other.shape)
            out._backward = _backward
            
        return out
    
    def __rsub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return other - self
    
    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, requires_grad=self.requires_grad or other.requires_grad, children=(self, other), op='*')
        if out.requires_grad:
            def _backward():
                if self.requires_grad:
                    self.grad += unbroadcast(other.data * out.grad, self.shape)
                if other.requires_grad:
                    other.grad += unbroadcast(self.data * out.grad, other.shape)
            out._backward = _backward 
        return out
        
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data / other.data,requires_grad=self.requires_grad or other.requires_grad,children=(self, other), op='/')
        
        if out.requires_grad:
            def _backward():
                if self.requires_grad:
                    self.grad += unbroadcast((1 / other.data) * out.grad, self.shape)
                if other.requires_grad:
                    other.grad += unbroadcast((-self.data / (other.data ** 2)) * out.grad, other.shape)
            out._backward = _backward
        
        return out
    
    def __rtruediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return other / self  # 反转调用
    
    def __pow__(self, power):
        power = power if isinstance(power, Tensor) else Tensor(power)
        out = Tensor(self.data ** power.data, requires_grad=self.requires_grad or power.requires_grad, children=(self, power), op='**')
        
        if out.requires_grad:
            def _backward():
                if self.requires_grad:
                    self.grad += unbroadcast((power.data * self.data ** (power.data - 1)) * out.grad, self.shape)
                if power.requires_grad:
                    safe_log = np.log(self.data + 1e-10)  # 防止 log(0)
                    power.grad += unbroadcast((self.data ** power.data) * safe_log * out.grad, power.shape)
            out._backward = _backward

        return out

    # def __matmul__(self, other):
    #     other = other if isinstance(other, Tensor) else Tensor(other)
    #     out = Tensor(self.data @ other.data, requires_grad=self.requires_grad or other.requires_grad, children=(self, other), op='matmul')
        
    #     if out.requires_grad:
    #         def _backward():
    #             if self.requires_grad:
    #                 grad = out.grad @ other.data.transpose((*range(other.data.ndim - 2), -1, -2))
    #                 while len(grad.shape) > len(other.data.shape):
    #                     grad = grad.sum(axis=0)
    #                 self.grad += grad
    #             if other.requires_grad:
    #                 grad = self.data.transpose((*range(self.data.ndim - 2), -1, -2)) @ out.grad
    #                 while len(grad.shape) > len(other.data.shape):
    #                     grad = grad.sum(axis=0)
    #                 other.grad += grad
    #         out._backward = _backward
        
    #     return out
    
    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out_data = self.data @ other.data  # 更明确的矩阵乘法接口
        out = Tensor(out_data, requires_grad=self.requires_grad or other.requires_grad,children=(self, other),op='matmul')

        if out.requires_grad:
            # 预计算转置维度（避免重复计算）
            self_axes = tuple(range(self.data.ndim - 2)) + (-1, -2)
            other_axes = tuple(range(other.data.ndim - 2)) + (-1, -2)
            
            def _backward():
                if self.requires_grad:
                    # 使用einsum替代手动降维，更高效且易读
                    grad = np.einsum('...ij,...jk->...ik', out.grad, other.data.transpose(other_axes))
                    # 处理广播维度
                    extra_dims = grad.ndim - self.data.ndim
                    if extra_dims > 0:
                        grad = grad.sum(axis=tuple(range(extra_dims)))
                    self.grad += grad
                
                if other.requires_grad:
                    grad = np.einsum('...ij,...jk->...ik', self.data.transpose(self_axes), out.grad)
                    extra_dims = grad.ndim - other.data.ndim
                    if extra_dims > 0:
                        grad = grad.sum(axis=tuple(range(extra_dims)))
                    other.grad += grad
            
            out._backward = _backward
        
        return out

    def sum(self, axis=None, keepdims=False):
        '''对 Tensor 的所有元素求和，返回标量 Tensor。'''
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims,dtype=np.float64), requires_grad=self.requires_grad, children=(self,), op='sum')  # 创建新的 Tensor，表示求和结果
        
        if self.requires_grad:
            def _backward():
                # ∂f(Σx)/∂x=∂f(Σx)/∂(Σx)*∂(Σx)/∂x=∂f(Σx)/∂(Σx)
                grad = out.grad
                if not keepdims and axis:
                    grad = np.expand_dims(grad, axis)
                self.grad += np.ones_like(self.data) * grad
            out._backward = _backward  # 将反向传播函数绑定到输出 Tensor
        
        return out  # 返回求和结果
    
    def maximum(self, other):
        '''求出两个 Tensor 之间的最大值'''
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(np.maximum(self.data, other.data,dtype=np.float64), requires_grad=self.requires_grad or other.requires_grad, children=(self, other), op='maximum')

        if out.requires_grad:
            def _backward():
                if self.requires_grad:
                    self.grad += (self.data >= other.data) * out.grad
                if other.requires_grad:
                    other.grad += (self.data < other.data) * out.grad
            out._backward = _backward

        return out

    def max(self, axis=None, keepdims=False):
        '''求出 Tensor 按某一维度的最大值'''
        out = Tensor(self.data.max(axis=axis, keepdims=keepdims), requires_grad=self.requires_grad, children = (self,), op = 'max')
        if self.requires_grad:
            def _backward():
                grad = out.grad
                if not keepdims and axis is not None:
                    grad = np.expand_dims(grad, axis=axis)
                max_vals = self.data.max(axis=axis, keepdims=True)
                mask = self.data == max_vals
                count = np.sum(mask, axis=axis, keepdims=True, dtype=np.float64)  # 对最大值均分梯度
                self.grad += mask * grad / count
            out._backward = _backward

        return out
    
    def exp(self):
        out = Tensor(np.exp(self.data), requires_grad=self.requires_grad, children=(self,), op='exp')
        
        if self.requires_grad:
            def _backward():
                    self.grad += out.data * out.grad
            out._backward = _backward

        return out
    
    def log(self):
        out = Tensor(np.log(self.data + 1e-10,dtype=np.float64), requires_grad=self.requires_grad, children=(self,), op='log')  # 防止 log(0)
        
        if self.requires_grad:
            def _backward():
                self.grad += (1 / (self.data + 1e-10)) * out.grad
            out._backward = _backward

        return out
    
    def abs(self):
        out = Tensor(np.abs(self.data, dtype=np.float64), requires_grad=self.requires_grad, children=(self,), op="abs")
        if self.requires_grad:
            def _backward():
                self.grad += np.sign(self.data) * out.grad
            out._backward = _backward
        
        return out

    def reshape(self, shape):
        '''
        返回新的 Tensor，其数据是 reshape 之后的（重排原数据），梯度反向传播会 reshape 回原来的形状。
        '''
        out_data = self.data.reshape(shape)
        out = Tensor(out_data, requires_grad=self.requires_grad, children=(self,), op='reshape')

        if self.requires_grad:
            def _backward():   
                # 把梯度 reshape 回原来的形状
                self.grad += out.grad.reshape(self.shape)
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
        
        # 维度对齐优化
        ndim = self.data.ndim
        pad_width = tuple(
            pad_width[i] if i < len(pad_width) else (0, 0)
            for i in range(ndim)
        )

        # 使用 np.pad 进行前向计算
        out_data = np.pad(self.data, pad_width, mode='constant')

        out = Tensor(out_data, requires_grad=self.requires_grad, children=(self,), op='pad')

        # 构造反向传播：裁剪掉 padding 区域
        slices = tuple(
            slice(p[0], p[0] + s) for p, s in zip(pad_width, self.shape)
        )
        if self.requires_grad:
            def _backward():
                grad = out.grad[slices]
                self.grad += grad

            out._backward = _backward
        return out
    
    def stack(tensors, axis=0):
        '''将一组 Tensor 沿指定维度拼接成一个新 Tensor。'''
        assert all(isinstance(t, Tensor) for t in tensors), "所有元素必须是Tensor"
        assert len(tensors) > 0, "输入Tensor列表不能为空"

        data = np.stack([t.data for t in tensors], axis=axis)
        requires_grad = any(t.requires_grad for t in tensors)
        out = Tensor(data, requires_grad=requires_grad, children=tuple(tensor for tensor in tensors), op='pad')
        
        if requires_grad:
            def _backward():
                grads = np.split(out.grad, len(tensors), axis=axis)
                for t, g in zip(tensors, grads):
                    if t.requires_grad:
                        t.grad += g.squeeze(axis=axis)
            out._backward = _backward
        
        return out

    def mean(self, axis=None, keepdims=False):
        data = self.data.mean(axis=axis, keepdims=keepdims)
        out = Tensor(data, requires_grad=self.requires_grad, children=(self,), op='mean')

        if self.requires_grad:
            axes = axis if isinstance(axis, tuple) else ((axis,) if axis is not None else None)

            def _backward():
                grad = out.grad

                # 保证 grad 是带有 keepdims 的形状（方便广播）
                if not keepdims and axes is not None:
                    shape = list(out.data.shape)
                    for ax in sorted(axes):
                        shape.insert(ax, 1)
                    grad = grad.reshape(shape)

                # 将梯度平均值广播回原始 shape
                scale = np.prod([self.data.shape[ax] for ax in axes]) if axes is not None else self.data.size
                grad = grad / scale
                np.add(self.grad, grad, out=self.grad, where=np.ones_like(self.data, dtype=bool))  # 原地加快一些

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
            self.grad = np.zeros_like(self.data, dtype=np.float64)
        else:
            self.grad = None

    def backward(self, remove_graph=True):
        ''' 反向传播 '''
        # 初始化 loss 的导数为 1
        self.grad =np.ones_like(self.data)

        # 拓扑排序
        topo: list[Tensor] = []
        visited = set()
        def build_topo(t: Tensor):
            if t not in visited and t.requires_grad:
                visited.add(t)
                for child in t._children:
                    build_topo(child)
                topo.append(t)
        build_topo(self)


        # 按拓扑顺序执行每个 tensor 的 _backward，开始反向传播
        for node in reversed(topo):
            node._backward()
            if (remove_graph):
                node._children = set()
                node._backward = lambda: None

    # def remove_graph(self):
        # # 拓扑排序
        # topo: list[Tensor] = []
        # visited = set()
        # def build_topo(t: Tensor):
        #     if t not in visited:
        #         visited.add(t)
        #         for child in t._children:
        #             build_topo(child)
        #         topo.append(t)
        # build_topo(self)

        # # 按拓扑顺序执行每个 tensor 的 _backward，开始反向传播
        # for node in reversed(topo):
        #     node._children = set()
        #     node._backward = lambda: None

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
    if grad.shape == shape:
        return grad
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

# def broadcast_to(self, shape):
    #     '''
    #     自动广播到目标形状，并在反向传播时正确累加梯度。
    #     '''
    #     out_data = np.broadcast_to(self.data, shape)
    #     out = Tensor(out_data, requires_grad=self.requires_grad, children=(self,), op='broadcast')

    #     def _backward():
    #         if self.requires_grad:
    #             grad = out.grad

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

    #             self.grad += grad

    #     out._backward = _backward
    #     return out

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