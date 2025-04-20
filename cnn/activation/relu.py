from cnn.core import tensor, Tensor
from cnn.base.layer import Layer

class ReLU(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x:Tensor)->Tensor:
        x = tensor.max(x, 0)
        return x