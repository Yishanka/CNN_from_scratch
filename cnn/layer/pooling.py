from cnn.base.layer import Layer
from cnn.core import Tensor
import numpy as np

class MaxPool2d(Layer):
    def __init__(self, kernel_size, stride=None, padding=0):
        """
        最大池化层
        
        参数:
            kernel_size: 池化窗口大小
            stride: 步长，默认与kernel_size相同
            padding: 填充大小，默认为0
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        
    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播
        
        参数:
            x: 输入张量，形状为(batch_size, channels, height, width)
            
        返回:
            池化后的张量
        """
        batch_size, channels, height, width = x.shape
        
        # 计算输出尺寸
        out_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        # 初始化输出
        output = np.zeros((batch_size, channels, out_height, out_width))
        
        # 对每个batch和channel应用池化操作
        for b in range(batch_size):
            for c in range(channels):
                for i in range(0, out_height):
                    for j in range(0, out_width):
                        h_start = i * self.stride - self.padding
                        w_start = j * self.stride - self.padding
                        h_end = min(h_start + self.kernel_size, height)
                        w_end = min(w_start + self.kernel_size, width)
                        h_start = max(0, h_start)
                        w_start = max(0, w_start)
                        
                        pool_region = x.data[b, c, h_start:h_end, w_start:w_end]
                        if pool_region.size > 0:  # 确保池化区域不为空
                            output[b, c, i, j] = np.max(pool_region)
        
        return Tensor(output, _children=(x,), _op='maxpool2d')

    def __repr__(self):
        return f"MaxPool2d(kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding})"


class AvgPool2d(Layer):
    def __init__(self, kernel_size, stride=None, padding=0):
        """
        平均池化层
        
        参数:
            kernel_size: 池化窗口大小
            stride: 步长，默认与kernel_size相同
            padding: 填充大小，默认为0
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        
    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播
        
        参数:
            x: 输入张量，形状为(batch_size, channels, height, width)
            
        返回:
            池化后的张量
        """
        batch_size, channels, height, width = x.shape
        
        # 计算输出尺寸
        out_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        # 初始化输出
        output = np.zeros((batch_size, channels, out_height, out_width))
        
        # 对每个batch和channel应用池化操作
        for b in range(batch_size):
            for c in range(channels):
                for i in range(0, out_height):
                    for j in range(0, out_width):
                        h_start = i * self.stride - self.padding
                        w_start = j * self.stride - self.padding
                        h_end = min(h_start + self.kernel_size, height)
                        w_end = min(w_start + self.kernel_size, width)
                        h_start = max(0, h_start)
                        w_start = max(0, w_start)
                        
                        pool_region = x.data[b, c, h_start:h_end, w_start:w_end]
                        if pool_region.size > 0:  # 确保池化区域不为空
                            output[b, c, i, j] = np.mean(pool_region)
        
        return Tensor(output, _children=(x,), _op='avgpool2d')

    def __repr__(self):
        return f"AvgPool2d(kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding})"