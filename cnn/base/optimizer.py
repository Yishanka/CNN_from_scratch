from cnn.core import Tensor, Parameter

class Optimizer:
    def __init__(self, lr):
        self.lr = lr

    def __call__(self, params: Parameter):
        """ 调用时直接执行 step """
        self.step(params)

    def step(self, params: Parameter)->Tensor:
        raise NotImplementedError("step 方法未实现")