import numpy as np
from cnn.core import Tensor


class Parameter(Tensor):
    '''
    Parameter 类继承自 Tensor，用于标识需要训练的可学习参数。默认 requires_grad=True。
    '''

    def __init__(self, shape: int | tuple):
        '''
        初始化 Parameter，默认初始化为全零张量。

        Parameters:
            shape: 数据的形状，可以是整数或元组。
        '''
        # 调用父类 Tensor 的初始化方法，传入全零数据和 requires_grad=True
        data = np.zeros(shape=shape)
        super().__init__(data, True)

        
    def xavier_uniform(self):
        a = np.sqrt(6.0 / sum(self.data.shape))
        self.data = np.random.uniform(-a, a, size=self.data.shape)

    def xavier_normal(self):
        sigma = np.sqrt(2.0 / sum(self.data.shape))
        self.data = np.random.normal(0, sigma, size=self.data.shape)

    def he_uniform(self):
        a = np.sqrt(6.0 / self.data.shape[1])
        self.data = np.random.uniform(-a, a, size=self.data.shape)

    def he_normal(self):
        sigma = np.sqrt(2.0 / self.data.shape[1])
        self.data = np.random.normal(0, sigma, size=self.data.shape)