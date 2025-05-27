from cnn.core import Parameter, Tensor
from cnn.base import Optimizer

class Adam(Optimizer):
    def __init__(self, lr=0.001, min_lr=1e-8, lr_decay = 1.0, beta1=0.9, beta2 = 0.999, eps=1e-8):
        '''
        Adam 优化器，基于一阶和二阶动量进行梯度下降
        '''
        super().__init__(lr, min_lr, lr_decay)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self._t = 0
        self.m = {}  # 一阶矩估计
        self.v = {}  # 二阶矩估计

    def _step(self, params: list[Parameter], lr):
        self._t += 1
        for i, param in enumerate(params):
            if i not in self.m:
                self.m[i] = Tensor.zeros_like(param)
                self.v[i] = Tensor.zeros_like(param)

            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * param.grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (param.grad ** 2)

            m_hat = self.m[i] / (1 - self.beta1 ** self._t)
            v_hat = self.v[i] / (1 - self.beta2 ** self._t)
            delta_grad = lr * m_hat / (v_hat**0.5 + self.eps)

            param.step(delta_grad) # step 不涉及 tensor 计算与矩阵构建