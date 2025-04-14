from cnn.core import Tensor
from cnn.nn import Parameter, Module
from cnn.utils.init_params import he_normal, zeros

class Linear(Module):
    def __init__(self, in_features: int, out_features: int, init_w=he_normal, init_b=zeros):
        self.weight = Parameter(init_w(in_features, out_features))
        self.bias = Parameter(init_b((out_features, 1)))

    def forward(self, x)->Tensor:
        return self.weight @ x + self.bias
    
    def parameter(self)->list:
        return [self.weight, self.bias]

if __name__ == '__main__':
    pass