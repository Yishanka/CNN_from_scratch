import numpy as np
from cnn.core import Tensor


class Parameter(Tensor):
    '''
    Parameter 类继承自 Tensor，用于标识需要训练的可学习参数。默认 requires_grad=True。
    '''

    def __init__(self, shape: int | tuple, is_reg = False):
        '''
        初始化 Parameter，默认初始化为全零张量。

        Parameters:
            shape: 数据的形状，可以是整数或元组。
        '''
        # 调用父类 Tensor 的初始化方法，传入全零数据和 requires_grad=True
        data = np.zeros(shape=shape)
        super().__init__(data, True)
        self.is_reg = is_reg
    
    def __repr__(self):
        return super().__repr__()
    
    def step(self, delta_grad: Tensor):
        '''
        Parameters:
            delta_grad(Tensor): 梯度减少的值 
        '''
        self._data -= delta_grad._data
