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
        self._data = np.array(data, dtype=np.float32) # 将数据转换为 numpy 数组
        self._grad = np.zeros_like(self._data) if requires_grad else None  # 梯度初始化为零
        self._children = set(_children)  # 子节点集合
        self._op = _op  # 操作符
        self._backward = lambda: None  # 反向传播的梯度计算函数，默认为空
        self.shape = self._data.shape
        self.size = self._data.size
        self.requires_grad = requires_grad  # 是否需要计算梯度

    @property
    def T(self):  
        '''对 Tensor: m*n 进行转置操作，返回 Tensor: n*m。'''
        out = Tensor(self._data.T, requires_grad=self.requires_grad, _children=(self,), _op='T')
        def _backward():
            if self.requires_grad:
                self._grad += out._grad.T
        out._backward = _backward
        return out
    
    def __repr__(self):
        return f"data：{self._data}\ngrad: {self._grad}"
    
    def __getitem__(self, idx):
        out = Tensor(self._data[idx], requires_grad=self.requires_grad, _children=(self,), _op='getitem')

        def _backward():
            if self.requires_grad:
                grad = np.zeros_like(self._data)
                np.add.at(grad, idx, out._grad)  # 这行是 NumPy 的广播安全反向写法
                self._grad += grad
        out._backward = _backward

        return out
    
    def __add__(self, other):
        '''重载加法运算符，支持两个 Tensor: m*n 相加，返回 Tensor: m*n。'''
        other = other if isinstance(other, Tensor) else Tensor(other)
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
        return self.__add__(other)
    
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
                self._grad += -out._grad
        out._backward = _backward
        return out
    
    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self.__add__(-other)
    
    def __mul__(self, other):
        '''重载乘法运算符，支持两个 Tensor: m*n 逐元素相乘，返回 Tensor: m*n。'''
        other = other if isinstance(other, Tensor) else Tensor(other)
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
    
    def __matmul__(self, other):
        '''重载矩阵乘法运算符，支持 Tensor: m*n 和 Tensor: n*p 进行矩阵乘法，返回 Tensor: m*p。'''
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self._data @ other._data, requires_grad=self.requires_grad or other.requires_grad, _children=(self, other), _op='matmul')
        def _backward():
            # ∂f(XY)/∂X=∂f(XY)/∂(XY)@∂(XY)/∂X=∂f(XY)/∂(XY)@Y.T
            if self.requires_grad:
                self._grad += out._grad @ other._data.T
            if other.requires_grad:
                other._grad += self._data.T @ out._grad
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
        '''求出两个张量之间的最大值，广播机制基于 numpy 规则'''
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
        '''求出某个张量的最大值'''
        out = Tensor(self._data.max(axis=axis, keepdims=keepdims), requires_grad=self.requires_grad, _children = (self,), _op = 'max')

        def _backward():
            max_vals = np.max(self._data, axis=axis, keepdims=True)
            mask = self._data == max_vals
            grad = mask * out._grad
            self._grad += grad
        out._backward = _backward

        return out
    
    def exp(self):
        out = Tensor(np.exp(self._data), requires_grad=self.requires_grad, _children=(self,), _op='exp')

        def _backward():
            self._grad += out._data * out._grad
        out._backward = _backward

        return out
    
    def log(self):
        out = Tensor(np.log(self._data + 1e-8), requires_grad=self.requires_grad, _children=(self,), _op='log')  # 防止 log(0)

        def _backward():
            self._grad += (1 / (self._data + 1e-8)) * out._grad
        out._backward = _backward

        return out
    
    def zero_grad(self):
        '''将梯度清零。'''
        if self.requires_grad:
            self._grad = np.zeros_like(self._data)

    # def one_grad(self):
    #     '''将梯度设为 1 '''
    #     if self.requires_grad:
    #         self.grad = np.ones_like(self.data)
    
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

if __name__ == '__main__':
    tensor_a = Tensor([[1,2,2],[1,2,2]], requires_grad=True)
    tensor_b = Tensor([[1,2,2],[1,2,2]], requires_grad=True)
    tensor = tensor_a - tensor_b
    tensor.sum().backward()
    print(tensor_b)