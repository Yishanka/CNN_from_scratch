from cnn.base.layer import Layer
from cnn.core import Parameter, Tensor

class Linear(Layer):
    def __init__(self, in_features: int, out_features: int):
        self.weight = Parameter((out_features, in_features))
        self.weight.he_normal()
        self.bias = Parameter((out_features, 1))

    def forward(self, x)->Tensor:
        return self.weight @ x + self.bias