from cnn.core import Tensor
from cnn.base.layer import Layer

class Softmax(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        # 数值稳定性
        shifted = x - x.max(axis=1, keepdims=True)  
        exp = shifted.exp()
        out = exp / exp.sum(axis=1, keepdims=True)
        return out
    

class ReLU(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x:Tensor)->Tensor:
        out = Tensor.maximum(x, 0)
        return out
    
class LeakyReLU(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x:Tensor)->Tensor:
        out = Tensor.maximum(x, 0.01*x)
        return out
    