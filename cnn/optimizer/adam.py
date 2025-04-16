from cnn.core import Tensor, Parameter
from cnn.base import Optimizer

class Adam(Optimizer):
    def __init__(self, lr):
        super().__init__(lr)
    
    def step(self, params:Parameter)->Tensor:
        pass

if __name__ == '__main__':
    pass