from cnn.core import Tensor
from cnn.base.layer import Layer

class ReLU(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x:Tensor)->Tensor:
        out = Tensor.max(x, 0)
        return out
    
class LeakyReLU(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x:Tensor)->Tensor:
        out = Tensor.max(x, 0.01*x)
        return out
    