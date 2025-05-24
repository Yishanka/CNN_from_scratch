from cnn.base.layer import Layer
from cnn.core import Tensor
import numpy as np

class Flatten(Layer):
    def __init__(self):
        '''
        扁平化层，将输入张量从形状(batch_size, channels, height, width)
        转换为(batch_size, channels * height * width)
        '''
        super().__init__()
    
    def _forward(self, x: Tensor) -> Tensor:
        '''
        前向传播
        
        Parameters:
            x: 输入张量，形状为(batch_size, channels, height, width)
            
        Returns:
            扁平化后的张量，形状为(batch_size, channels * height * width)
        '''
        batch_size = x.shape[0]
        output = x.reshape((batch_size, -1))
        return output