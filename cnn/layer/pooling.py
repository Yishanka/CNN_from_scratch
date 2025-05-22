from cnn.base.layer import Layer
from cnn.core import Tensor

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
        self._padding = padding if isinstance(padding, tuple) else (padding, padding)
        
    def _forward(self, x: Tensor) -> Tensor:
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
        cols = im2col(x, kernel_size=(kh, kw), stride=(sh, sw), padding=(ph, pw))  # [bs, c, oh*ow, kh*kw]

        # === 找最大值及其索引（用于反向传播）===
        max_vals = cols.max(axis=-1)  # [bs, c, oh*ow]

        # reshape 输出为 [B, C, OH, OW]
        out_h = (h + 2 * ph - kh) // sh + 1
        out_w = (w + 2 * pw - kw) // sw + 1
        out = max_vals.reshape((bs, c, out_h, out_w))
        return out

def im2col(x: Tensor, kernel_size, stride, padding) -> Tensor:
    # x: [bs, ic, h, w]
    bs, ic, h, w = x.shape
    kh, kw = kernel_size
    sh, sw = stride
    ph, pw = padding

    if ph > 0 or pw > 0:
        x = x.pad(((0, 0), (0, 0), (ph, ph), (pw, pw)))  # -> [bs, ic, h+2ph, w+2pw]

    _, _, h_padded, w_padded = x.shape
    oh = (h_padded - kh) // sh + 1
    ow = (w_padded - kw) // sw + 1

    # 获取 stride 步长
    s0, s1, s2, s3 = x.data.strides
    
    # 构造窗口的 as_strided 视图，维度为 [bs, ic, oh, ow, kh, kw]
    shape = (bs, ic, oh, ow, kh, kw)
    strides = (s0, s1, s2 * sh, s3 * sw, s2, s3)
    
    x = x.as_strided(shape, strides)
    # x.as_strided_inplace(shape, strides)
    # reshape 为 [bs, ic, oh*ow, kh*kw]
    x = x.reshape((bs, ic, oh * ow, kh * kw))
    return x

# def im2col(x: Tensor, kernel_size, stride, padding) -> Tensor:
#     # x: [batch_size, in_channels, height, weight]
#     _, _, h, w = x.shape
#     kh, kw = kernel_size
#     sh, sw = stride
#     ph, pw = padding

#     # padding
#     if ph > 0 or pw > 0:
#         x = x.pad(((0, 0), (0, 0), (ph, ph), (pw, pw)))  # [bs, ic, h+2ph, w+2pw]

#     oh = (h + 2 * ph - kh) // sh + 1
#     ow = (w + 2 * pw - kw) // sw + 1

#     cols = []
#     for i in range(oh):
#         for j in range(ow):
#             patch = x[:, :, i*sh:i*sh+kh, j*sw:j*sw+kw]  # [bs, ic, kh, kw]
#             cols.append(patch.reshape((x.shape[0], x.shape[1], -1))) # [bs, ic, kh*kw]
#     return Tensor.stack(cols, axis=2)  # [bs, ic, oh*ow, kh*kw]
