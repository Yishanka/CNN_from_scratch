from cnn.base.layer import Layer
from cnn.core import Parameter, Tensor
import numpy as np

class BatchNorm2d(Layer):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        """
        二维批量归一化层
        
        参数:
            num_features: 输入特征的通道数
            eps: 为了数值稳定性而添加到分母的值，默认1e-5
            momentum: 用于运行平均值计算的动量因子，默认0.1
        """
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        # 可学习参数
        self.gamma = Parameter((num_features,))  # 缩放参数
        self.beta = Parameter((num_features,))   # 平移参数
        
        # 初始化参数值
        self.gamma.data = np.ones(num_features)
        self.beta.data = np.zeros(num_features)
        
        # 用于推理阶段的运行时统计
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
        
        # 训练模式标志
        self.training = True
    
    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播
        
        参数:
            x: 输入张量，形状为(batch_size, num_features, height, width)
            
        返回:
            归一化后的张量
        """
        # 获取输入形状
        batch_size, channels, height, width = x.shape
        
        # 重塑数据以便于计算统计量: (N, C, H, W) -> (N*H*W, C)
        x_reshaped = x.data.transpose(0, 2, 3, 1).reshape(-1, channels)
        
        if self.training:
            # 计算批量均值和方差
            batch_mean = np.mean(x_reshaped, axis=0)
            batch_var = np.var(x_reshaped, axis=0) + self.eps
            
            # 更新运行时统计值
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var
            
            # 归一化
            x_norm = (x_reshaped - batch_mean) / np.sqrt(batch_var)
            
            # 使用当前批次的统计量
            mean_used = batch_mean
            var_used = batch_var
        else:
            # 使用运行时统计量进行归一化
            x_norm = (x_reshaped - self.running_mean) / np.sqrt(self.running_var + self.eps)
            mean_used = self.running_mean
            var_used = self.running_var
        
        # 缩放和平移
        out = self.gamma.data * x_norm + self.beta.data
        
        # 重塑回原始形状: (N*H*W, C) -> (N, C, H, W)
        out = out.reshape(batch_size, height, width, channels).transpose(0, 3, 1, 2)
        
        # 创建输出Tensor
        return Tensor(
            out,
            _children=(x, self.gamma, self.beta),
            _op='batchnorm2d'
        )
    
    def train(self):
        """设置为训练模式"""
        self.training = True
        
    def eval(self):
        """设置为评估模式"""
        self.training = False
        
    def __repr__(self):
        return f"BatchNorm2d(num_features={self.num_features}, eps={self.eps}, momentum={self.momentum})"