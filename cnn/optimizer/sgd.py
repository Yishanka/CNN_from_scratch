from cnn.base import Optimizer
from cnn.core import Parameter

class SGD(Optimizer):
    def __init__(self, lr=0.001, min_lr=1e-8, lr_decay=0.999):
        super().__init__(lr, min_lr, lr_decay)

    def _step(self, params: list[Parameter], lr):
        for param in params:
            delta_grad = lr * param.grad
            param.step(delta_grad)
            