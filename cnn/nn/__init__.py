from cnn.core import Tensor

class Parameter(Tensor):
    def __init__(self, data, requires_grad = True):
        super().__init__(data, requires_grad)
        
class Module:
    def parameters(self):
        params = []
        for attr in self.__dict__.values():
            if isinstance(attr, Parameter):
                params.append(attr)
            elif isinstance(attr, Module):
                params.extend(attr.parameters())
        return params
    
from cnn.nn.convolution import Conv2d
from cnn.nn.linear import Linear