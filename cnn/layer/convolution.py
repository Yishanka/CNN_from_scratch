from cnn.base.layer import Layer
from cnn.core import Parameter, Tensor, he_normal

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

        # 获取输入的信息，x shape: (batch_size, in_channels, in_height, in_width)
        batch_size, in_channels, in_height, in_width = x.shape

        # 获取输出的信息，便于后续编码
        out_channels, weight_in_channels, kernel_height, kernel_width = self._weight.shape
        stride_height, stride_width = self._stride
        pad_height, pad_width = self._padding

        # 保证无意外输入，防御性编程，可纳入到 Tensor 的计算中
        if in_channels != weight_in_channels:
            raise KeyError('输入通道数不符合卷积层定义')
        
        # 1. padding
        if pad_height > 0 or pad_width > 0:
            x = x.pad(((0, 0), (0, 0), (pad_height, pad_height), (pad_width, pad_width)))

        # 2. 计算输出尺寸并初始化输出 Tensor
        out_height = (in_height + 2 * pad_height - kernel_height) // stride_height + 1
        out_width = (in_width + 2 * pad_width - kernel_width) // stride_width + 1
        output = Tensor.zeros((batch_size, out_channels, out_height, out_width), requires_grad=self._weight.requires_grad)

        # 3. 遍历所有输出通道、位置，构造输出
        output_slices = []

        for i in range(out_height):
            row_slices = []
            for j in range(out_width):
                h_start = i * stride_height
                h_end = h_start + kernel_height
                w_start = j * stride_width
                w_end = w_start + kernel_width

                x_patch = x[:, :, h_start:h_end, w_start:w_end]  # [Batch_size, in_channels, kernel_height, kernel_weight] 
                out = Tensor.sum(x_patch * self._weight, axis=(1, 2, 3)) + self._bias

                row_slices.append(out)  # shape: [B, C_out]
            
            # 每一行输出特征图，拼成 [B, C_out, W]
            output_slices.append(Tensor.stack(row_slices, axis=2))  # shape: [B, C_out, H, W]

        # 最终 shape: [B, C_out, H, W]
        return Tensor.stack(output_slices, axis=2)