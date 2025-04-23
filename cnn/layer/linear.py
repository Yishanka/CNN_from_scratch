from cnn.core import Parameter, Tensor
from cnn.base.layer import Layer
import numpy as np

class Linear(Layer):
    def __init__(self, in_features: int, out_features: int, bias=True):
        """
        全连接层
        
        参数:
            in_features: 输入特征数
            out_features: 输出特征数
            bias: 是否使用偏置项，默认True
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias_flag = bias
        
        # 初始化权重参数
        self.weight = Parameter((out_features, in_features))
        self.weight.he_normal()  # 使用He初始化
        
        # 初始化偏置参数
        if bias:
            self.bias = Parameter((out_features, 1))
            self.bias.data = np.zeros((out_features, 1))
        else:
            self.bias = None
    
    def forward(self, x: Tensor) -> Tensor:
        """前向传播"""
        # 处理输入: 如果输入是二维的，将其视为batch_size x features
        if len(x.shape) == 2:
            # 检查输入特征维度
            if x.shape[1] != self.in_features:
                raise ValueError(f"输入特征维度 {x.shape[1]} 与层的输入维度 {self.in_features} 不匹配")
                
            # 转置输入使其兼容矩阵乘法: (batch_size, in_features) -> (in_features, batch_size)
            x_T = x.T()
            output = self.weight @ x_T
            
            # 添加偏置（如果有）
            if self.bias_flag:
                output = output + self.bias
                
            # 转置回原来的形状: (out_features, batch_size) -> (batch_size, out_features)
            return output.T()
        
        else:
            # 如果是多维输入，先展平为(batch_size, features)
            batch_size = x.shape[0]
            flattened = x.data.reshape(batch_size, -1)
            x_flat = Tensor(flattened)
            
            # 使用展平后的数据递归调用forward
            return self.forward(x_flat)
    
    def __repr__(self):
        return f"Linear(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias_flag})"