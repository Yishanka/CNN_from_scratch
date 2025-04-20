from cnn.base import Optimizer
from cnn.core import Parameter

class SGD(Optimizer):
    def __init__(self, lr=0.001):
        super().__init__()
        self._lr = lr

    def _step(self, params: list[Parameter]):
        for param in params:
            param.step(self._lr * param.grad)