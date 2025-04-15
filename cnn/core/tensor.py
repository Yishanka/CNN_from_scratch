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
        self.data = np.array(data, dtype=np.float32) # 将数据转换为 numpy 数组
        self.requires_grad = requires_grad  # 是否需要计算梯度
        self._children = set(_children)  # 子节点集合
        self._op = _op  # 操作符
        self.grad = np.zeros_like(self.data) if requires_grad else None  # 梯度初始化为零
        self._backward = lambda: None  # 反向传播的梯度计算函数，默认为空
    
    def __repr__(self):
        return f"{self.data}"
    
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

    def sum(self):
        '''对 Tensor 的所有元素求和，返回标量 Tensor。'''
        out = Tensor(self.data.sum(), requires_grad=self.requires_grad, _children=(self,), _op='sum')  # 创建新的 Tensor，表示求和结果
        def _backward():
            # ∂f(Σx)/∂x=∂f(Σx)/∂(Σx)*∂(Σx)/∂x=∂f(Σx)/∂(Σx)
            if self.requires_grad:
                self.grad += np.ones_like(self.data) * out.grad  # 更新 self 的梯度
        out._backward = _backward  # 将反向传播函数绑定到输出 Tensor
        return out  # 返回求和结果

    def T(self):  
        '''对 Tensor: m*n 进行转置操作，返回 Tensor: n*m。'''
        out = Tensor(self.data.T, requires_grad=self.requires_grad, _children=(self,), _op='T')
        def _backward():
            if self.requires_grad:
                self.grad += out.grad.T
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
    
    def backward(self):
        ''' 反向传播接口，计算梯度。要求当前 Tensor 为标量。 '''
        assert self.data.size == 1, "只能对标量调用 backward"
        self.grad = np.ones_like(self.data) # 对自己的导数为1，作为链式法则的起点
        topo, visited = [], set() # 拓扑排序的容器
        # 递归构建拓扑排序
        def build_topo(t: Tensor):
            ''''''
            if t not in visited:
                visited.add(t)
                for child in t._children:
                    build_topo(child)
                topo.append(t)
        build_topo(self)  # 从当前节点开始构建拓扑排序，建立反向传播的节点顺序
        for node in reversed(topo):
            node._backward()

    def zero_grad(self):
        '''将梯度清零。'''
        if self.requires_grad:
            self.grad = np.zeros_like(self.data)

if __name__ == '__main__':
    tensor_a = Tensor([[1,2,2],[1,2,2]], requires_grad=True)
    tensor_b = Tensor([[1,2,2],[1,2,2]], requires_grad=True)
    tensor = tensor_a + tensor_b
    print(tensor.data)
    loss = tensor.sum()
    loss.backward()