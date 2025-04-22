from cnn.core import tensor, Tensor
from cnn.base.layer import Layer

class ReLU(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, X:Tensor)->Tensor:
        out = tensor.max(X, 0)
        return out
    
class LeakyReLU(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, X:Tensor)->Tensor:
        out = tensor.max(X, 0.01*X)
        return out
    