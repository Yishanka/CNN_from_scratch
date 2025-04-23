import numpy as np

class Tensor:
    '''
    Tensor 类，用于表示多维数组，并支持自动求导。
    '''
    def __init__(self, data, requires_grad:bool=False, _children=(), _op:str='', _backward=lambda: None):
        '''
        Parameters:
            data: 数据，转换为 numpy 数组
            requires_grad: 是否需要计算梯度，默认 False
            _children: 子节点，用于构建计算图，默认空元组
            _op: 操作符，用于标识该 Tensor 是如何生成的，默认空字符串
            _backward: 反向传播的梯度计算函数
        '''
        self.data = np.array(data, dtype=np.float32) # 将数据转换为 numpy 数组
        self.requires_grad = requires_grad  # 是否需要计算梯度
        self._children = set(_children)  # 子节点集合
        self._op = _op  # 操作符
        self.grad = np.zeros_like(self.data) if requires_grad else None  # 梯度初始化为零
        self._backward = _backward  # 反向传播的梯度计算函数
    
    def __repr__(self):
        return f"{self.data}"
    
    @property
    def shape(self):
        '''返回数据的形状'''
        return self.data.shape
    
    def __add__(self, other):
        '''重载加法运算符，支持两个 Tensor: m*n 相加，返回 Tensor: m*n。'''
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            self.data + other.data, 
            requires_grad=self.requires_grad or other.requires_grad, 
            _children=(self, other), 
            _op='+'
        )
        def _backward():
            # ∂f(x+y)/∂x=∂f(x+y)/∂(x+y)*∂(x+y)/∂x=∂f(x+y)/∂(x+y)
            if self.requires_grad:
                self.grad += out.grad
            if other.requires_grad:
                other.grad += out.grad
        out._backward = _backward
        return out
    
    def __mul__(self, other):
        '''重载乘法运算符，支持两个 Tensor: m*n 逐元素相乘，返回 Tensor: m*n。'''
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            self.data * other.data, 
            requires_grad=self.requires_grad or other.requires_grad, 
            _children=(self, other), 
            _op='*'
        )
        def _backward():
            # ∂f(xy)/∂x=∂f(xy)/∂(xy)*∂(xy)/∂x=∂f(xy)/∂(xy)*y
            if self.requires_grad:
                self.grad += other.data * out.grad
            if other.requires_grad:
                other.grad += self.data * out.grad
        out._backward = _backward 
        return out
    
    def __neg__(self):
        """重载负号运算符，返回张量的负值"""
        return self * (-1)
    
    def __sub__(self, other):
        """重载减法运算符，返回两个张量的差"""
        return self + (-other)
    
    def __truediv__(self, other):
        """重载除法运算符，返回两个张量的商"""
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            self.data / other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
            _op='/'
        )
        def _backward():
            if self.requires_grad:
                self.grad += out.grad / other.data
            if other.requires_grad:
                other.grad += -self.data / (other.data * other.data) * out.grad
        out._backward = _backward
        return out
    
    def __pow__(self, power):
        """重载幂运算符，返回张量的幂"""
        assert isinstance(power, (int, float)), "幂必须是标量"
        out = Tensor(
            self.data ** power,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op=f'**{power}'
        )
        def _backward():
            if self.requires_grad:
                self.grad += (power * self.data ** (power - 1)) * out.grad
        out._backward = _backward
        return out

    def sum(self, axis=None, keepdims=False):
        '''对 Tensor 的元素求和，支持沿指定轴求和'''
        out_data = self.data.sum(axis=axis, keepdims=keepdims)
        out = Tensor(out_data, requires_grad=self.requires_grad, _children=(self,), _op='sum')  
        def _backward():
            if self.requires_grad:
                # 创建梯度的广播版本
                if axis is not None:
                    grad_shape = list(self.data.shape)
                    if not keepdims:
                        if isinstance(axis, int):
                            grad_shape[axis] = 1
                        else:
                            for a in sorted(axis, reverse=True):
                                grad_shape[a] = 1
                    grad_broadcast = np.ones(grad_shape) * out.grad
                    self.grad += np.broadcast_to(grad_broadcast, self.data.shape)
                else:
                    self.grad += np.ones_like(self.data) * out.grad
        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        '''对 Tensor 的元素求平均值，支持沿指定轴求平均值'''
        # 计算元素总数
        num_elements = np.prod(self.shape) if axis is None else np.prod(np.array(self.shape)[np.array(axis)])
        
        # 使用sum实现mean
        return self.sum(axis=axis, keepdims=keepdims) / num_elements

    def T(self):
        '''对 Tensor: m*n 进行转置操作，返回 Tensor: n*m。'''
        out = Tensor(self.data.T, requires_grad=self.requires_grad, _children=(self,), _op='T')
        def _backward():
            if self.requires_grad:
                self.grad += out.grad.T
        out._backward = _backward
        return out
    
    def exp(self):
        '''计算 Tensor 各元素的指数 e^x'''
        out = Tensor(np.exp(self.data), requires_grad=self.requires_grad, _children=(self,), _op='exp')
        def _backward():
            if self.requires_grad:
                self.grad += out.data * out.grad
        out._backward = _backward
        return out
    
    def log(self):
        '''计算 Tensor 各元素的自然对数 ln(x)'''
        out = Tensor(np.log(self.data), requires_grad=self.requires_grad, _children=(self,), _op='log')
        def _backward():
            if self.requires_grad:
                self.grad += out.grad / self.data
        out._backward = _backward
        return out
    
    def max(self, axis=None, keepdims=False):
        '''获取 Tensor 的最大值，支持沿指定轴获取最大值'''
        max_vals = np.max(self.data, axis=axis, keepdims=keepdims)
        out = Tensor(max_vals, requires_grad=self.requires_grad, _children=(self,), _op='max')
        def _backward():
            if self.requires_grad:
                # 创建与原始张量相同形状的梯度矩阵
                grad = np.zeros_like(self.data)
                
                # 对于每个最大值位置，设置相应的梯度
                if axis is not None:
                    # 获取最大值位置的布尔掩码
                    mask = (self.data == np.max(self.data, axis=axis, keepdims=True))
                    # 计算每行/列最大值元素的数量
                    counts = np.sum(mask, axis=axis, keepdims=True)
                    # 将梯度均分给所有最大值元素
                    grad = np.where(mask, out.grad / counts, 0)
                else:
                    # 全局最大值
                    mask = (self.data == np.max(self.data))
                    count = np.sum(mask)
                    grad = np.where(mask, out.grad / count, 0)
                
                self.grad += grad
        out._backward = _backward
        return out
    
    def abs(self):
        '''计算 Tensor 各元素的绝对值'''
        out = Tensor(np.abs(self.data), requires_grad=self.requires_grad, _children=(self,), _op='abs')
        def _backward():
            if self.requires_grad:
                self.grad += np.sign(self.data) * out.grad
        out._backward = _backward
        return out
    
    def __matmul__(self, other):
        '''重载矩阵乘法运算符，支持 Tensor: m*n 和 Tensor: n*p 进行矩阵乘法，返回 Tensor: m*p。'''
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data, requires_grad=self.requires_grad or other.requires_grad, _children=(self, other), _op='matmul')
        def _backward():
            # ∂f(XY)/∂X=∂f(XY)/∂(XY)@∂(XY)/∂X=∂f(XY)/∂(XY)@Y.T
            if self.requires_grad:
                self.grad += out.grad @ other.data.T
            if other.requires_grad:
                other.grad += self.data.T @ out.grad
        out._backward = _backward
        return out
    
    def relu(self):
        '''对 Tensor 应用 ReLU 激活函数。'''
        out = Tensor(np.maximum(0, self.data), requires_grad=self.requires_grad, _children=(self,), _op='relu')
        def _backward():
            if self.requires_grad:
                self.grad += (self.data > 0) * out.grad
        out._backward = _backward
        return out

    def zero_grad(self):
        '''将梯度清零。'''
        if self.requires_grad:
            self.grad = np.zeros_like(self.data)

    def one_grad(self):
        '''将梯度设为 1 '''
        if self.requires_grad:
            self.grad = np.ones_like(self.data)
            
    def backward(self):
        """手动调用的反向传播函数"""
        # 初始化loss的导数为1
        self.one_grad()
        
        # 拓扑排序
        topo = []
        visited = set()
        def build_topo(t: "Tensor"):
            if t not in visited:
                visited.add(t)
                for child in t._children:
                    build_topo(child)
                topo.append(t)
        build_topo(self)
        
        # 按拓扑顺序执行每个tensor的_backward，开始反向传播
        for node in reversed(topo):
            node._backward()
            
    def item(self):
        """获取单个元素张量的值"""
        assert self.data.size == 1, "只能对单个元素的Tensor调用item()"
        return float(self.data)

if __name__ == '__main__':
    tensor_a = Tensor([[1,2,2],[1,2,2]], requires_grad=True)
    tensor_b = Tensor([[1,2,2],[1,2,2]], requires_grad=True)
    tensor = tensor_a + tensor_b
    print(tensor.data)
    loss = tensor.sum()
    loss.backward()