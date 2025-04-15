from cnn.core import Tensor
from cnn.nn.layer import Layer
class Model:
    '''
    '''
    def parameters(self):
        '''
        初始化所有模型的参数，递归调用所有子层的初始化函数
        Returns:
            params (list): 所有可训练参数组成的列表
        '''
        params = []
        for attr in self.__dict__.values():
            if isinstance(attr, Layer):
                params.extend(attr.parameters())
        return params
    
    def forward(self, x)->Tensor:
        for attr in self.__dict__.values():
            if isinstance(attr, Layer):
                x = attr.forward(x)
        return x

if __name__ == '__main__':
    pass