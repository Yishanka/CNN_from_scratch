import numpy as np

class Tensor:
    def __init__(self, data, requires_grad:bool=False, _children=(), _op:str=''):
        '''
        Args:
            data: 数据，转换为 numpy 数组
            requires_grad: 是否需要计算梯度，默认 False
            _children: 子节点，用于构建计算图，默认空元组
            _op: 操作符，用于标识该 Tensor 是如何生成的，默认空字符串
        '''
        self.data = np.array(data, dtype=np.float32)
        self.requires_grad = requires_grad
        self._children = set(_children)
        self._op = _op
        self.grad = np.zeros_like(self.data) if requires_grad else None
        def _backward():
            pass
        self._backward = _backward # 反向传播的梯度计算函数，基于链式法则确定
    
    # 重载加法运算符 R^n*m -> R^n*m
    def __add__(self, other):
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
    
    # 重载乘法运算符 R^n*m -> R^n*m
    def __mul__(self, other):
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

    # 张量元素求和 R^n*m -> R
    def sum(self):
        out = Tensor(self.data.sum(), requires_grad=self.requires_grad, _children=(self,), _op='sum')  # 创建新的 Tensor，表示求和结果
        def _backward():
            # ∂f(Σx)/∂x=∂f(Σx)/∂(Σx)*∂(Σx)/∂x=∂f(Σx)/∂(Σx)
            if self.requires_grad:
                self.grad += np.ones_like(self.data) * out.grad  # 更新 self 的梯度
        out._backward = _backward  # 将反向传播函数绑定到输出 Tensor
        return out  # 返回求和结果

    # 转置 R^n*m -> R^m*n
    def T(self):  
        out = Tensor(self.data.T, requires_grad=self.requires_grad, _children=(self,), _op='T')
        def _backward():
            if self.requires_grad:
                self.grad += out.grad.T
        out._backward = _backward
        return out
    
    # 矩阵乘法 R^n*m, R^m*p -> R^n*p
    def __matmul__(self, other):
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
    
    # relu 激活函数, R^n*m -> R^n*m
    def relu(self):
        out = Tensor(np.maximum(0, self.data), requires_grad=self.requires_grad, _children=(self,), _op='relu')
        def _backward():
            if self.requires_grad:
                self.grad += (self.data > 0) * out.grad
        out._backward = _backward
        return out
    
    # 反向传播接口，从此处开始，逐节点计算梯度
    def backward(self):
        assert self.data.size == 1, "只能对标量调用 backward" # 从最终的张量开始求导，最终张量必须是标量
        self.grad = np.ones_like(self.data) # 对自己的导数为1
        topo, visited = [], set() # 类中的容器
        def build_topo(t: Tensor):
            if t not in visited:
                visited.add(t)
                for child in t._children:
                    build_topo(child)
                topo.append(t)
        build_topo(self) # 从起点开始拓扑排序，建立反向传播的节点顺序
        for node in reversed(topo):
            node._backward()

    # 梯度清零
    def zero_grad(self):
        if self.requires_grad:
            self.grad = np.zeros_like(self.data)

if __name__ == '__main__':
    tensor_a = Tensor([[1,2,2],[1,2,2]], requires_grad=True)
    tensor_b = Tensor([[1,2,2],[1,2,2]], requires_grad=True)
    tensor = tensor_a + tensor_b
    print(tensor.data)
    loss = tensor.sum()
    loss.backward()