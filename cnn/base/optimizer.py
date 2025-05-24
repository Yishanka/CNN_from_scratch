from cnn.core import Parameter

class Optimizer:
    def __init__(self, lr=1e-3, min_lr = 1e-8, lr_decay = 0.999):
        '''
        Optimizer 基类
        '''
        self.lr = lr
        self.min_lr = min_lr
        self.lr_decay = lr_decay
        self.step_count = 0

    def __call__(self, params: list[Parameter]):
        self.step(params)

    def step(self, params: list[Parameter]):
        lr = self.lr * (self.lr_decay ** (self.step_count)) + self.min_lr
        self._step(params, lr)
        self.step_count += 1

    def _step(self, params: list[Parameter], lr):
        raise NotImplementedError("子类必须实现 _step 方法")
