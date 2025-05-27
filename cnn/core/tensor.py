import numpy as np
from numpy.lib.stride_tricks import as_strided
class Tensor:
    '''
    Tensor 类，用于表示多维数组，并支持自动求导。
    '''
    def __init__(self, data, requires_grad:bool=False, children=(), op: str=None, dtype=np.float32):
        '''
        Parameters:
            data: 数据，转换为 numpy 数组
            requires_grad: 是否需要计算梯度，默认 False
            _children: 子节点，用于构建计算图，默认空元组
            dtype: 数据类型
        '''
        self.requires_grad = requires_grad  # 是否需要计算梯度
        self.data = np.asarray(data, dtype=dtype)
        self.grad = np.zeros_like(self.data, dtype=dtype) if self.requires_grad else None  # 梯度初始化为零
        self._children = tuple(children) if self.requires_grad else tuple() # 子节点集合
        self._op = op  # 操作符
        self._backward = lambda: None  # 反向传播的梯度计算函数，默认为空
        self.dtype = np.float32
    
# <======= 核心方法：反向传播 =======>
    def backward(self, remove_graph=True):
        ''' 反向传播 '''
        # 初始化 loss 的导数为 1
        self.grad =np.ones_like(self.data)

        # 拓扑排序
        topo: list[Tensor] = []
        visited = set()
        def build_topo(t: Tensor):
            if t not in visited and t.requires_grad: # 不需要求梯度的 Tensor 不需要进入计算图
                visited.add(t)
                for child in t._children:
                    build_topo(child)
                topo.append(t)
        build_topo(self)

        # 按拓扑顺序执行每个 tensor 的 _backward，开始反向传播
        for node in reversed(topo):
            node._backward()
            if (remove_graph):
                node._children = tuple()
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
    
    def zero_grad(self):
        self.grad = np.zeros_like(self.data, dtype=self.dtype) if self.requires_grad else None

# <======= 属性操作 =======>
    @property
    def shape(self):
        return self.data.shape
    
    @property
    def size(self):
        return self.data.size

    @property
    def T(self):
        out = Tensor(self.data.T, requires_grad=self.requires_grad, children=(self,), op='T')
        
        if self.requires_grad:
            def _backward():
                np.add(self.grad, out.grad.T, out=self.grad)  
            out._backward = _backward
        
        return out
    
# <======= 特殊方法重载 =======>
    def __repr__(self):
        data_str = f'{self.data:.4f}' if np.isscalar(self.data) else np.array2string(
            self.data, precision=4, separator=', ', suppress_small=True)

        if self.requires_grad:
            grad_str = f'{self.grad:.4f}' if np.isscalar(self.grad) else np.array2string(
                self.grad, precision=4, separator=', ', suppress_small=True)
        else:
            grad_str = 'None'

        return (f'data:\n{data_str}\n'
                f'grad:\n{grad_str}')
    
    def __getitem__(self, idx: tuple):
        idx = idx if isinstance(idx, tuple) else tuple(idx)
        out = Tensor(self.data[idx], requires_grad=self.requires_grad, children=(self,), op='getitem')
        if self.requires_grad:
            def _backward():
                np.add.at(self.grad, idx, out.grad)
            out._backward = _backward

        return out
    
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, requires_grad=self.requires_grad or other.requires_grad, children=(self, other), op='+')

        if out.requires_grad:
            def _backward():
                if self.requires_grad:
                    np.add(self.grad, unbroadcast(out.grad, self.shape), out=self.grad)
                if other.requires_grad:
                    np.add(other.grad, unbroadcast(out.grad, other.shape), out=other.grad)
            out._backward = _backward
            
        return out

    def __radd__(self, other):
        return self + other

    def __neg__(self):
        out = Tensor(-self.data, requires_grad=self.requires_grad, children=(self,), op='neg')

        if self.requires_grad:
            def _backward():
                np.add(self.grad, -out.grad, out=self.grad)
            out._backward = _backward

        return out
    
    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data - other.data, requires_grad=self.requires_grad or other.requires_grad, children=(self, other), op='-')

        if out.requires_grad:
            def _backward():
                if self.requires_grad:
                    np.add(self.grad, unbroadcast(out.grad, self.shape),out=self.grad)
                if other.requires_grad:
                    np.add(other.grad, -unbroadcast(out.grad, other.shape), out=other.grad)
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
                    np.add(self.grad, unbroadcast(other.data * out.grad, self.shape), out=self.grad)
                if other.requires_grad:
                    np.add(other.grad, unbroadcast(self.data * out.grad, other.shape), out=other.grad)
            out._backward = _backward 
        return out
        
    def __rmul__(self, other):
        return self * other
    
    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data / other.data,requires_grad=self.requires_grad or other.requires_grad,children=(self, other), op='/')
        if out.requires_grad:
            def _backward():
                if self.requires_grad:
                    np.add(self.grad, unbroadcast((1 / other.data) * out.grad, self.shape), out=self.grad)
                if other.requires_grad:
                    np.add(other.grad, unbroadcast((-self.data / (other.data ** 2)) * out.grad, other.shape), out=other.grad)
            out._backward = _backward
        
        return out
    
    def __rtruediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return other / self
    
    def __pow__(self, power):
        power = power if isinstance(power, Tensor) else Tensor(power)
        out = Tensor(self.data ** power.data, requires_grad=self.requires_grad or power.requires_grad, children=(self, power), op='**')
        
        if out.requires_grad:
            def _backward():
                if self.requires_grad:
                    np.add(self.grad, unbroadcast((power.data * self.data ** (power.data - 1)) * out.grad, self.shape), out=self.grad)
                if power.requires_grad:
                    safe_log = np.log(self.data + 1e-10)  # 防止 log(0)
                    np.add(power.grad, unbroadcast((self.data ** power.data) * safe_log * out.grad, power.shape),out=power.grad)
            out._backward = _backward

        return out
   
    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(np.matmul(self.data, other.data), requires_grad=self.requires_grad or other.requires_grad, children=(self, other), op='@')

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
                    np.add(self.grad, grad, out=self.grad)
                
                if other.requires_grad:
                    grad = np.einsum('...ij,...jk->...ik', self.data.transpose(self_axes), out.grad)
                    extra_dims = grad.ndim - other.data.ndim
                    if extra_dims > 0:
                        grad = grad.sum(axis=tuple(range(extra_dims)))
                    np.add(other.grad, grad, out=other.grad)
            
            out._backward = _backward
        
        return out

# <======= numpy 部分计算重载 =======>
    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims,dtype=self.dtype), requires_grad=self.requires_grad, children=(self,), op='sum')
        
        if self.requires_grad:
            def _backward():
                grad = np.expand_dims(out.grad, axis) if not keepdims and axis else out.grad
                np.add(self.grad, grad, out=self.grad)
            out._backward = _backward 
        
        return out
    
    def exp(self):
        out = Tensor(np.exp(self.data, dtype=self.dtype), requires_grad=self.requires_grad, children=(self,), op='exp')
        
        if self.requires_grad:
            def _backward():
                np.add(self.grad, out.data * out.grad, out=self.grad)
            out._backward = _backward

        return out
    
    def log(self):
        out = Tensor(np.log(self.data + 1e-10, dtype=self.dtype), requires_grad=self.requires_grad, children=(self,), op='log')  # 防止 log(0)
        
        if self.requires_grad:
            def _backward():
                np.add(self.grad, (1 / (self.data + 1e-10)) * out.grad, out=self.grad)
            out._backward = _backward

        return out
    
    def abs(self):
        out = Tensor(np.abs(self.data, dtype=self.dtype), requires_grad=self.requires_grad, children=(self,), op='abs')
        if self.requires_grad:
            def _backward():
                np.add(self.grad, np.sign(self.data) * out.grad, out=self.grad)
            out._backward = _backward
        
        return out
    
    def mean(self, axis=None, keepdims=False):
        out = Tensor(self.data.mean(axis=axis, keepdims=keepdims, dtype=self.dtype), requires_grad=self.requires_grad, children=(self,), op='mean')

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

    def var(self, axis=None, keepdims=False, ddof=1):
        # 1. 直接计算均值（不创建Tensor节点）
        mean_data = np.mean(self.data, axis=axis, keepdims=True)
        
        # 2. 计算平方差和
        centered = self.data - mean_data
        squared = centered * centered
        sum_squared = np.sum(squared, axis=axis, keepdims=keepdims)
        
        # 3. 计算除数（考虑ddof）
        if axis is None:
            n = self.data.size
        else:
            n = np.prod([self.data.shape[ax] for ax in (axis if isinstance(axis, tuple) else (axis,))])
        out_data = sum_squared / (n - ddof)
        
        # 4. 构建输出Tensor
        out = Tensor(out_data, requires_grad=self.requires_grad, children=(self,), op='var')

        if self.requires_grad:
            def _backward():
                grad = out.grad
                
                # 处理keepdims广播
                if not keepdims and axis is not None:
                    shape = list(out.data.shape)
                    for ax in sorted(axis if isinstance(axis, tuple) else (axis,)):
                        shape.insert(ax, 1)
                    grad = grad.reshape(shape)
                
                # 方差的反向传播公式: d(var)/dx = 2*(x - mean)/(n - ddof)
                scale = 2.0 / (n - ddof)
                grad = grad * scale * (self.data - mean_data)
                
                # 累加梯度
                np.add(self.grad, grad, out=self.grad, where=np.ones_like(self.data, dtype=bool))

            out._backward = _backward

        return out
     

    def maximum(self, other):
        '''求出两个 Tensor 之间的最大值'''
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(np.maximum(self.data, other.data, dtype=self.dtype), requires_grad=self.requires_grad or other.requires_grad, children=(self, other), op='maximum')

        if out.requires_grad:
            def _backward():
                if self.requires_grad:
                    np.add(self.grad, unbroadcast((self.data >= other.data) * out.grad, self.shape), out= self.grad)
                if other.requires_grad:
                    np.add(other.grad, unbroadcast((self.data < other.data) * out.grad, other.shape), out=other.grad)
            out._backward = _backward

        return out

    def max(self, axis=None, keepdims=False):
        '''求出 Tensor 按某一维度的最大值'''
        out = Tensor(self.data.max(axis=axis, keepdims=keepdims), requires_grad=self.requires_grad, children = (self,), op='max')
        if self.requires_grad:
            max_vals = self.data.max(axis=axis, keepdims=True)
            mask = self.data == max_vals
            count = np.sum(mask, axis=axis, keepdims=True)  # 对最大值均分梯度
            def _backward():
                grad = out.grad
                if not keepdims and axis is not None:
                    grad = np.expand_dims(grad, axis=axis)
                np.add(self.grad, mask * grad / count, out=self.grad)
            out._backward = _backward

        return out

# <======= 张量形状改变操作 =======>
    
    def transpose(self, axes: tuple, inplace=False):
        if inplace:
            return Tensor._transpose_inplace(self, axes)
        else:
            return Tensor._transpose(self, axes)
        
    def _transpose_inplace(self, axes: tuple):
        '''通过置换维度来转置张量。'''
        self.data = self.data.transpose(axes)
        self.grad = self.grad.transpose(axes) if self.requires_grad else None
        old_backward = self._backward

        if self.requires_grad:
            inverse_axes = tuple(np.argsort(axes))
            def _backward():
                self.data = self.data.transpose(inverse_axes)
                self.grad = self.grad.transpose(inverse_axes)
                if old_backward:
                    old_backward()
            self._backward = _backward
            
            return self
        
    def _transpose(self, axes: tuple):
        '''通过置换维度来转置张量。'''
        out = Tensor(self.data.transpose(axes), requires_grad=self.requires_grad, children=(self,), op='transpose')

        if self.requires_grad:
            inverse_axes = tuple(np.argsort(axes))

            def _backward():
                np.add(self.grad, out.grad.transpose(inverse_axes), out=self.grad)
            out._backward = _backward
        
        return out
      
    def reshape(self, shape, inplace=False):
        '''重排数据，支持原地操作和新张量创建'''
        if inplace:
            return Tensor._reshape_inplace(self, shape)
        else:
            return Tensor._reshape(self, shape)

    def _reshape(self, shape):
        '''
        返回新的 Tensor，其数据是 reshape 之后的（重排原数据），梯度反向传播会 reshape 回原来的形状。
        '''
        out_data = self.data.reshape(shape)
        out = Tensor(out_data, requires_grad=self.requires_grad, children=(self,), op='reshape')

        if self.requires_grad:
            def _backward():   
                # 把梯度 reshape 回原来的形状
                np.add(self.grad, out.grad.reshape(self.shape), out=self.grad)
            out._backward = _backward
        
        return out
    
    def _reshape_inplace(self, shape):
        '''
        原地重排数据，梯度反向传播会 reshape 回原来的形状。
        '''
        old_shape = self.shape
        old_backward = self._backward
        self.data = self.data.reshape(shape)
        self.grad = self.grad.reshape(shape) if self.requires_grad else None

        if self.requires_grad:
            def _backward():   
                # 把梯度 reshape 回原来的形状
                self.grad = self.grad.reshape(old_shape)
                self.data = self.data.reshape(old_shape)
                if old_backward:
                    old_backward()
            self._backward = _backward
        return self
    
    def pad(self, pad_width: tuple[tuple[int, int], ...]):
        '''
        通用 zero-padding
        Parameters:
            pad_width: 与 np.pad 一致，例如：
                ((0, 0), (0, 0), (1, 2), (3, 3)) 表示：
                - 第三维前 pad 1，后 pad 2
                - 第四维前 pad 3，后 pad 3
        '''
        # 安全性校验
        assert all(isinstance(p, tuple) and len(p) == 2 for p in pad_width), '每个维度必须是 (before, after)'
        
        # 维度对齐优化
        ndim = self.data.ndim
        pad_width = tuple(
            pad_width[i] if i < len(pad_width) else (0, 0)
            for i in range(ndim)
        )

        out = Tensor(np.pad(self.data, pad_width, mode='constant'), requires_grad=self.requires_grad, children=(self,), op='pad')

        if self.requires_grad:
            slices = tuple(
                slice(p[0], p[0] + s) for p, s in zip(pad_width, self.shape)
            )
            def _backward():
                np.add(self.grad, out.grad[slices], out=self.grad)
            out._backward = _backward
        return out
    
    def stack(tensors, axis=0):
        '''将一组 Tensor 沿指定维度拼接成一个新 Tensor。'''
        assert all(isinstance(t, Tensor) for t in tensors), '所有元素必须是Tensor'
        assert len(tensors) > 0, '输入Tensor列表不能为空'

        requires_grad = any(t.requires_grad for t in tensors)
        out = Tensor(data = np.stack([t.data for t in tensors], axis=axis), requires_grad=requires_grad, children=tuple(tensor for tensor in tensors), op='stack')
        
        if requires_grad:
            def _backward():
                grads = np.split(out.grad, len(tensors), axis=axis)
                for t, g in zip(tensors, grads):
                    if t.requires_grad:
                        np.add(t.grad, g.squeeze(axis=axis), out=t.grad)
            out._backward = _backward
        
        return out

# <======= 视图操作 =======>    
    def as_strided(self, shape, strides, inplace=False):
        if inplace:
            Tensor._as_strided_inplace(self, shape, strides)
        else:
            return Tensor._as_strided(self, shape, strides)

    def _as_strided(self, shape, strides):
        '''创建视图'''
        out = Tensor(np.lib.stride_tricks.as_strided(self.data, shape=shape, strides=strides), requires_grad=self.requires_grad, children=(self,), op='as_strided')

        if self.requires_grad:
            grad_view = np.lib.stride_tricks.as_strided(self.grad, shape=shape, strides=strides)
            def _backward():
                np.add.at(grad_view, ..., out.grad)
            out._backward = _backward

        return out

    def _as_strided_inplace(self, shape, strides):
        '''原地创建视图，不创建新结点，减少分配内存'''

        # 保存原本的 backward 方法（可能是 None 或已有计算图）
        old_backward = self._backward
        
        grad_raw = self.grad
        # 修改 data 为视图
        self.data = np.lib.stride_tricks.as_strided(self.data, shape=shape, strides=strides)
        self.grad = np.lib.stride_tricks.as_strided(self.grad, shape=shape, strides=strides)

        if self.requires_grad:
            def _backward():
                self.grad = grad_raw
                if old_backward:
                    old_backward()

            self._backward = _backward

        return self
    
    # def broadcast_to(self, shape):
    #     '''
    #     自动广播到目标形状，并在反向传播时正确累加梯度。
    #     '''
    #     out_data = np.broadcast_to(self.data, shape)
    #     out = Tensor(out_data, requires_grad=self.requires_grad, children=(self,))
    #     if self.requires_grad:
    #         def _backward():
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

    #         out._backward = _backward
    #     return out

# <======= 张量创建操作 =======>
    def zeros(shape, requires_grad=False, dtype=np.float32):
        '''创建一个给定 shape 的零张量。'''
        return Tensor(np.zeros(shape, dtype=dtype), requires_grad=requires_grad)

    def zeros_like(data, dtype=np.float32):
        '''返回一个全零张量，shape 保持一致，不参与反向传播'''
        if isinstance(data, Tensor):
            data = data.data
        return Tensor(np.zeros_like(data, dtype=dtype))
    
    def ones(shape, requires_grad=False, dtype=np.float32):
        '''创建一个给定 shape 的全一张量。'''
        return Tensor(np.ones(shape,dtype=dtype), requires_grad=requires_grad)
    
# <======= 其他操作 =======>
    def argmax(self, axis=None, keepdims=False):
        '''不参与求导计算'''
        out = self.data.argmax(axis=axis, keepdims=keepdims)
        return out
    
    def to_int(self):
        '''原地修改数据类型到int, 不参与梯度下降'''
        self.data = np.int32(self.data)

# <======= 逆广播函数 =======> 
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

# <=========== 外部操作，暂时不使用 ===========>
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
    pass