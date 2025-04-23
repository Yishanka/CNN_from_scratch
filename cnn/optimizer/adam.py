from cnn.core import Tensor, Parameter
from cnn.base.optimizer import Optimizer
import numpy as np

class Adam(Optimizer):
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        """
        Adam优化器
        
        参数:
            params: 需要优化的参数列表
            lr: 学习率，默认0.001
            betas: 用于计算梯度及其平方的运行平均值的系数，默认(0.9, 0.999)
            eps: 为了数值稳定性而添加到分母中的项，默认1e-8
        """
        super().__init__(lr)
        self.params = params
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.m = {}  # 一阶矩估计
        self.v = {}  # 二阶矩估计
        self.t = 0  # 时间步
        
        # 初始化动量和方差累积器
        for i, param in enumerate(self.params):
            self.m[i] = np.zeros_like(param.data)
            self.v[i] = np.zeros_like(param.data)
    
    def step(self):
        """执行一步参数更新"""
        self.t += 1
        
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue
                
            # 获取参数的梯度
            grad = param.grad
            
            # 更新偏差校正的一阶矩估计
            self.m[i] = self.betas[0] * self.m[i] + (1 - self.betas[0]) * grad
            
            # 更新偏差校正的二阶矩估计
            self.v[i] = self.betas[1] * self.v[i] + (1 - self.betas[1]) * (grad * grad)
            
            # 计算偏差校正后的估计
            m_corrected = self.m[i] / (1 - self.betas[0] ** self.t)
            v_corrected = self.v[i] / (1 - self.betas[1] ** self.t)
            
            # 更新参数
            param.data = param.data - self.lr * m_corrected / (np.sqrt(v_corrected) + self.eps)
    
    def zero_grad(self):
        """清除所有参数的梯度"""
        for param in self.params:
            if param.grad is not None:
                param.zero_grad()
                
    def __repr__(self):
        return f"Adam(lr={self.lr}, betas={self.betas}, eps={self.eps})"

if __name__ == '__main__':
    pass