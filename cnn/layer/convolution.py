from cnn.base.layer import Layer
from cnn.core import Parameter, Tensor

class Conv2d:
    def __init__(
        self,
        in_channels: int, 
        out_channels: int, 
        kernel_size: int, 
        stride: int,
        padding: bool,
        bias: bool
    ):
        pass

    def forward(self, x:Tensor):
        pass
