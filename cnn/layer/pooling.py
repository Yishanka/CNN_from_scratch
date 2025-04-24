from cnn.base.layer import Layer
from cnn.core import Tensor
from cnn.layer.utils import im2col

class MaxPool2d(Layer):
    def __init__(self, kernel_size, stride=None, padding=0):
        """
        最大池化层
        
        Parameters:
            kernel_size: 池化窗口大小
            stride: 步长，默认与kernel_size相同
            padding: 填充大小，默认为0
        """
        super().__init__()        
        stride = stride if stride is not None else kernel_size
        self._kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self._stride = stride if isinstance(stride, tuple) else (stride, stride)
        self._padding = padding if isinstance(stride, tuple) else (stride, stride)
        
    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters:
            x: 输入张量，形状为(batch_size, channels, height, width)
            
        Returns:
            池化后的张量
        """
        kh, kw = self._kernel_size
        sh, sw = self._stride
        ph, pw = self._padding

        bs, c, h, w = x.shape

        # === 使用 im2col 展开 ===
        cols = im2col(x, kernel_size=(kh, kw), stride=(sh, sw), padding=(ph, pw))  # [B, OH*OW, C*kh*kw]

        # === 找最大值及其索引（用于反向传播）===
        max_vals = cols.max(axis=2)  # [B, OH*OW]

        # reshape 输出为 [B, C, OH, OW]
        out_h = (h + 2 * ph - kh) // sh + 1
        out_w = (w + 2 * pw - kw) // sw + 1
        out = max_vals.reshape((bs, c, out_h, out_w))

        return out

    def __repr__(self):
        return f"MaxPool2d(kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding})"


# class AvgPool2d(Layer):
#     def __init__(self, kernel_size, stride=None, padding=0):
#         """
#         平均池化层
        
#         参数:
#             kernel_size: 池化窗口大小
#             stride: 步长，默认与kernel_size相同
#             padding: 填充大小，默认为0
#         """
#         super().__init__()
#         self.kernel_size = kernel_size
#         self.stride = stride if stride is not None else kernel_size
#         self.padding = padding
        
#     def forward(self, x: Tensor) -> Tensor:
#         """
#         前向传播
        
#         Parameters:
#             x: 输入张量，形状为(batch_size, channels, height, width)
            
#         Returns:
#             池化后的张量
#         """
#         batch_size, channels, height, width = x.shape
        
#         # 计算输出尺寸
#         out_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
#         out_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1
        
#         # 初始化输出
#         output = Tensor.zeros((batch_size, channels, out_height, out_width))
        
#         # 对每个batch和channel应用池化操作
#         for b in range(batch_size):
#             for c in range(channels):
#                 for i in range(0, out_height):
#                     for j in range(0, out_width):
#                         h_start = i * self.stride - self.padding
#                         w_start = j * self.stride - self.padding
#                         h_end = min(h_start + self.kernel_size, height)
#                         w_end = min(w_start + self.kernel_size, width)
#                         h_start = max(0, h_start)
#                         w_start = max(0, w_start)
                        
#                         pool_region = x[b, c, h_start:h_end, w_start:w_end]
#                         if pool_region.size > 0:  # 确保池化区域不为空
#                             output[b, c, i, j] = Tensor.mean(pool_region)
        
#         return Tensor(output, _children=(x,), _op='avgpool2d')