from cnn.base.layer import Layer
from cnn.core import Parameter, Tensor
import numpy as np

class BatchNorm2d(Layer):
    def __init__(self, channels, momentum=0.1):
        """
        二维批量归一化层
        
        参数:
            num_features: 输入特征的通道数
            momentum: 用于运行平均值计算的动量因子，默认0.1
        """
        super().__init__()
        self.num_features = channels
        self.eps = 1e-8
        self.momentum = momentum
        
        # 可学习参数
        self.gamma = Parameter((channels,), is_reg=False)  # 缩放参数
        self.beta = Parameter((channels,), is_reg=False)   # 平移参数
        
        # 初始化参数值
        self.gamma= Tensor.ones(channels)
        self.beta = Tensor.zeros(channels)
        
        # 用于推理阶段的运行时统计
        self.running_mean = Tensor.zeros(channels)
        self.running_var = Tensor.ones(channels)
        
        # 训练模式标志
        self.training = True
    
    def _forward(self, x: Tensor) -> Tensor:
        """
        前向传播
        
        Parameters:
            x: 输入张量，形状为(batch_size, num_features, height, width)
            
        Returns:
            归一化后的张量
        """
        # 获取输入形状
        batch_size, channels, height, width = x.shape
        
        # 重塑数据以便于计算统计量: (bs, c, h, w) -> (bs*h*w, c)
        x_reshaped = x.transpose((0, 2, 3, 1)).reshape((-1, channels))
        
        if self.training:
            # 计算批量均值和方差
            batch_mean = x_reshaped.mean(axis=0)
            batch_var = x_reshaped.var(axis=0) + self.eps
            
            # 更新运行时统计值，脱离计算图
            self.running_mean.data = (1 - self.momentum) * self.running_mean.data + self.momentum * batch_mean.data
            self.running_var.data = (1 - self.momentum) * self.running_var.data + self.momentum * batch_var.data
                        
            # 归一化
            x_norm = (x_reshaped - batch_mean) / (batch_var**0.5)
            
        else:
            # 使用运行时统计量进行归一化
            x_norm = (x_reshaped - self.running_mean) / (self.running_var + self.eps) ** 0.5
        
        # 缩放和平移
        out = x_norm * self.gamma + self.beta
        
        # 重塑回原始形状: (N*H*W, C) -> (N, C, H, W)
        out = out.reshape((batch_size, height, width, channels)).transpose((0, 3, 1, 2))
        
        return out