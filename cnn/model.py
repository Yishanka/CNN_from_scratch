from cnn.core import Tensor
from cnn.base import Layer, Loss, Optimizer
class Model:
    def __init__(self):
        self._layers = []
        self._loss = None
        self._optimizer = None
        self._params = []

    def __setattr__(self, name, value):
        if isinstance(value, Layer):
            self._layers.append(value)
            self._params.extend(value.parameters())
        elif isinstance(value, Loss):
            self._loss = value
        elif isinstance(value, Optimizer):
            self._optimizer = value
        super().__setattr__(name, value)

    def forward(self, x)->Tensor:
        for layer in self._layers:
            x = layer.forward(x)
        return x
    
    def compute_loss(self, pred, true)->Tensor:
        return self._loss(pred, true)
    
    def backward(self):
        self._loss.backward()

    def step(self):
        self._optimizer(self._params)

    def zero_grad(self):
        for layer in self._layers:
            layer.zero_grad()

if __name__ == '__main__':
    pass