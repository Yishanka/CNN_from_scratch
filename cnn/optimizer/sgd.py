from cnn.base import Optimizer
from cnn.core import Parameter

class SGD(Optimizer):
    def __init__(self, lr=0.001, min_lr=1e-8, decay_weight=0.99):
        super().__init__(lr, min_lr, decay_weight)

    def _step(self, params: list[Parameter], lr):
        for param in params:
            delta_grad = lr * param.grad
            param.step(delta_grad)
            