from cnn.core import Parameter, Tensor
from cnn.nn.init import he_normal

class Layer:
    def __init__(self):
        self.params = []

    def forward(self, x)->Tensor:
        return x
    
    def parameters(self)->list:
        return []
    

class Linear(Layer):
    def __init__(self, in_features: int, out_features: int):
        self.weight = he_normal(Parameter((out_features, in_features)))
        self.bias = Parameter((out_features, 1))

    def forward(self, x)->Tensor:
        return self.weight @ x + self.bias
    
    def parameters(self)->list:
        return [self.weight, self.bias]
    


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


if __name__ == '__main__':
    pass