from cnn.core import Tensor
from cnn.base.layer import Layer

class ReLU(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x:Tensor)->Tensor:
        x = x.maximum(0)
        return x