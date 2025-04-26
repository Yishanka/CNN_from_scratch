from cnn.core import Parameter, Tensor
from cnn.base import Optimizer

class Adam(Optimizer):
    def __init__(self, lr=0.001, beta1=0.9, beta2 = 0.999, eps=1e-8):
        super().__init__()
        self._lr = lr
        self._beta1 = beta1
        self._beta2 = beta2
        self._eps = eps
        self._t = 0
        self.m = {}  # 一阶矩估计
        self.v = {}  # 二阶矩估计
    
    def _step(self, params: list[Parameter]):
        self._t += 1
        for i, param in enumerate(params):
            if i not in self.m:
                self.m[i] = Tensor.zeros_like(param)
                self.v[i] = Tensor.zeros_like(param)

            self.m[i] = self._beta1 * self.m[i] + (1 - self._beta1) * param.grad
            self.v[i] = self._beta2 * self.v[i] + (1 - self._beta2) * (param.grad ** 2)

            m_hat = self.m[i] / (1 - self._beta1 ** self._t)
            v_hat = self.v[i] / (1 - self._beta2 ** self._t)
            delta_grad = self._lr * m_hat / (v_hat**0.5 + self._eps)

            param.step(delta_grad) # step 不涉及 tensor 计算与矩阵构建
            delta_grad.remove_graph()