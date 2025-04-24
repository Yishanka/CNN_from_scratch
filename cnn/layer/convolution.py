from cnn.base.layer import Layer
from cnn.core import Parameter, Tensor, he_normal
from cnn.layer.utils import im2col

class Conv2d(Layer):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple|int, stride: tuple|int=1, padding: tuple|int=0):
        '''
        初始化卷积层。
        Parameters:
            in_channels(int): 输入的通道数，如灰度图为 1，RGB 图为 3，以其他卷积层输出为输入的通道数为它的 out_channels
            out_channels(int): 输出的通道数，也是卷积核 kernel 的个数，控制输出特征图的个数
            kernel_size(tuple|int): 卷积核的大小
            stride(tuple|int): 卷积核移动的步长，默认为1
        '''
        super().__init__()
        # 处理输入
        self._padding = padding if isinstance(stride, tuple) else (stride, stride)
        self._stride = stride if isinstance(stride, tuple) else (stride, stride)

        # 定义卷积核，共 out_channels 个，每个处理 in_channels 张图，卷积核尺寸为 kh, kw
        kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        kh, kw = kernel_size
        self._weight = Parameter((out_channels, in_channels, kh, kw), is_reg=True)
        self._bias = Parameter((out_channels, 1), is_reg=False)
        he_normal(self._weight)

    def _forward(self, x: Tensor) -> Tensor:
        if not isinstance(x, Tensor):
            x = Tensor(x)

        bs, c, h, w = x.shape
        oc, ic, kh, kw = self._weight.shape
        sh, sw = self._stride
        ph, pw = self._padding

        if c != ic:
            raise ValueError("输入通道数与卷积核不匹配")

        # im2col 展开
        cols = im2col(x, (kh, kw), (sh, sw), (ph, pw))  # [bs, oh*ow, ic*kh*kw]
        weight_flat = (self._weight.reshape((oc, -1))).T  # [ic*kh*kw, oc]
        
        # 执行批量矩阵乘法
        out = cols @ weight_flat  # [bs, oh*ow, oc]
        out = out.transpose((0, 2, 1))  # [bs, oc, oh*ow]

        # 添加 bias 并 reshape 成卷积输出
        oh = (h + 2 * ph - kh) // sh + 1
        ow = (w + 2 * pw - kw) // sw + 1
        out = out + self._bias  # broadcast bias
        return out.reshape((bs, oc, oh, ow))