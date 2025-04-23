from cnn.base.layer import Layer
from cnn.core import Parameter, Tensor
import numpy as np

class Conv2d(Layer):
    def __init__(
        self,
        in_channels: int, 
        out_channels: int, 
        kernel_size: int, 
        stride: int = 1,
        padding: int = 0,
        bias: bool = True
    ):
        """
        二维卷积层
        
        参数:
            in_channels: 输入通道数
            out_channels: 输出通道数
            kernel_size: 卷积核大小，假设正方形卷积核
            stride: 卷积步长，默认为1
            padding: 零填充大小，默认为0
            bias: 是否使用偏置，默认为True
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias_flag = bias
        
        # 初始化卷积核权重 (out_channels, in_channels, kernel_size, kernel_size)
        self.weight = Parameter((out_channels, in_channels, kernel_size, kernel_size))
        # 使用He初始化
        scale = np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.weight.data = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * scale
        
        # 初始化偏置
        if bias:
            self.bias = Parameter((out_channels,))
            self.bias.data = np.zeros(out_channels)
        else:
            self.bias = None
    
    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播计算卷积
        
        参数:
            x: 输入张量，形状为(batch_size, in_channels, height, width)
            
        返回:
            输出特征图，形状为(batch_size, out_channels, out_height, out_width)
        """
        batch_size, _, height, width = x.shape
        
        # 计算输出特征图的尺寸
        out_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        # im2col变换
        col = self._im2col(x.data)
        
        # 重塑卷积核权重为矩阵乘法形式
        w_col = self.weight.data.reshape(self.out_channels, -1)
        
        # 执行矩阵乘法
        out = w_col @ col
        
        # 添加偏置（如果有）
        if self.bias_flag:
            out = out + self.bias.data.reshape(-1, 1)
        
        # 重塑输出为(batch_size, out_channels, out_height, out_width)
        out = out.reshape(self.out_channels, batch_size, out_height, out_width)
        out = out.transpose(1, 0, 2, 3)
        
        # 创建输出Tensor，设置反向传播函数
        return Tensor(
            out, 
            _children=(x, self.weight) if not self.bias_flag else (x, self.weight, self.bias),
            _op='conv2d'
        )
    
    def _im2col(self, x_data):
        """
        将输入数据转换为列形式，用于高效卷积计算
        
        参数:
            x_data: 输入数据的numpy数组
            
        返回:
            转换后的列数据
        """
        batch_size, in_channels, height, width = x_data.shape
        
        # 计算输出特征图的尺寸
        out_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        # 创建包含padding的输入数据
        if self.padding > 0:
            padded = np.zeros((batch_size, in_channels, height + 2 * self.padding, width + 2 * self.padding))
            padded[:, :, self.padding:self.padding+height, self.padding:self.padding+width] = x_data
        else:
            padded = x_data
        
        # 初始化输出列矩阵
        col = np.zeros((in_channels * self.kernel_size * self.kernel_size, batch_size * out_height * out_width))
        
        # 填充列矩阵
        col_idx = 0
        for b in range(batch_size):
            for i in range(0, height + 2 * self.padding - self.kernel_size + 1, self.stride):
                for j in range(0, width + 2 * self.padding - self.kernel_size + 1, self.stride):
                    # 提取当前patch
                    patch = padded[b, :, i:i+self.kernel_size, j:j+self.kernel_size]
                    col[:, col_idx] = patch.reshape(-1)
                    col_idx += 1
        
        return col
    
    def __repr__(self):
        return f"Conv2d(in_channels={self.in_channels}, out_channels={self.out_channels}, kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, bias={self.bias_flag})"

if __name__ == '__main__':
    pass