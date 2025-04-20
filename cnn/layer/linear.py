from cnn.core import Parameter, Tensor
from cnn.base.layer import Layer

class Linear(Layer):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self._weight = Parameter((in_features, out_features), is_reg=True)
        self._bias = Parameter((1, out_features), is_reg=False) # 已有广播机制，无需获取 batch_size
        self._weight.he_normal()

    def _forward(self, X)->Tensor:
        if not isinstance(X, Tensor):
            X = Tensor(X)
        bias = self._bias.repeat(axis=0, repeats=X.shape[0])
        out = X @ self._weight + bias
        return out