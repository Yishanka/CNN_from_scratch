from cnn.base.layer import Layer
from cnn.core import Tensor
import numpy as np

class Flatten(Layer):
    def __init__(self):
        """
        扁平化层，将输入张量从形状(batch_size, channels, height, width)
        转换为(batch_size, channels * height * width)
        """
        super().__init__()
    
    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播
        
        Parameters:
            x: 输入张量，形状为(batch_size, channels, height, width)
            
        Returns:
            扁平化后的张量，形状为(batch_size, channels * height * width)
        """
        batch_size = x.shape[0]
        flattened_shape = (batch_size, -1)
        output = x.reshape(flattened_shape)
        
        return Tensor(output, _children=(x,), _op='flatten')

    def __repr__(self):
        return "Flatten()"