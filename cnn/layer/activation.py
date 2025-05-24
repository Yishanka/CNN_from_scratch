from cnn.core import Tensor
from cnn.base.layer import Layer

class Softmax(Layer):
    def __init__(self):
        super().__init__()

    def _forward(self, x: Tensor) -> Tensor:
        # 数值稳定性
        shifted = x - x.max(axis=1, keepdims=True)  
        exp = shifted.exp()
        out = exp / exp.sum(axis=1, keepdims=True)
        return out
    
class ReLU(Layer):
    def __init__(self):
        super().__init__()

    def _forward(self, x:Tensor)->Tensor:
        out = x.maximum(x)
        return out
    
class LeakyReLU(Layer):
    def __init__(self):
        super().__init__()

    def _forward(self, x:Tensor)->Tensor:
        out = x.maximum(0.01*x)
        return out
    