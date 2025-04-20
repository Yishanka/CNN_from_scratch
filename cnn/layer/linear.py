from cnn.core import Parameter, Tensor
from cnn.base.layer import Layer

class Linear(Layer):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = Parameter((out_features, in_features))
        self.bias = Parameter((out_features, 1))
        self.weight.he_normal()

    def _forward(self, X)->Tensor:
        if not isinstance(X, Tensor):
            X = Tensor(X)
        out = self.weight @ X.T + self.bias
        return out.T