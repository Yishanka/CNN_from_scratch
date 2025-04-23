from cnn.base.layer import Layer
from cnn.core import Tensor
import numpy as np

class ReLU(Layer):
    def __init__(self):
        """ReLU激活函数: f(x) = max(0, x)"""
        super().__init__()
    
    def forward(self, x: Tensor) -> Tensor:
        """前向传播"""
        output = np.maximum(0, x.data)
        return Tensor(output, _children=(x,), _op='relu')
    
    def __repr__(self):
        return "ReLU()"


class Sigmoid(Layer):
    def __init__(self):
        """Sigmoid激活函数: f(x) = 1 / (1 + exp(-x))"""
        super().__init__()
    
    def forward(self, x: Tensor) -> Tensor:
        """前向传播"""
        output = 1 / (1 + np.exp(-x.data))
        return Tensor(output, _children=(x,), _op='sigmoid')
    
    def __repr__(self):
        return "Sigmoid()"


class Tanh(Layer):
    def __init__(self):
        """Tanh激活函数: f(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))"""
        super().__init__()
    
    def forward(self, x: Tensor) -> Tensor:
        """前向传播"""
        output = np.tanh(x.data)
        return Tensor(output, _children=(x,), _op='tanh')
    
    def __repr__(self):
        return "Tanh()"


class Softmax(Layer):
    def __init__(self, dim=1):
        """
        Softmax激活函数
        
        参数:
            dim: 执行softmax的维度，默认为1
        """
        super().__init__()
        self.dim = dim
    
    def forward(self, x: Tensor) -> Tensor:
        """前向传播"""
        # 为了数值稳定性，减去每行的最大值
        x_max = np.max(x.data, axis=self.dim, keepdims=True)
        exp_x = np.exp(x.data - x_max)
        output = exp_x / np.sum(exp_x, axis=self.dim, keepdims=True)
        return Tensor(output, _children=(x,), _op='softmax')
    
    def __repr__(self):
        return f"Softmax(dim={self.dim})"