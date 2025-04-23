from cnn.core import Tensor, Parameter
class Layer:
    def __init__(self):
        '''初始化 layer 对象，设置参数列表'''
        self._params = []

    def __setattr__(self, name, value):
        '''创建派生类对象时收集派生类的所有参数，放到参数列表中'''
        if isinstance(value, Parameter):
            self._params.append(value)
        super().__setattr__(name, value)
        
    def forward(self, x)->Tensor:
        '''前向传播函数，需要在派生类里面实现'''
        raise NotImplementedError("forward 方法未实现")
    
    def __call__(self, x)->Tensor:
        '''调用层时执行前向传播'''
        return self.forward(x)
    
    def zero_grad(self):
        '''将层中参数的梯度归零'''
        for param in self._params:
            param.zero_grad()

    def parameters(self):
        '''返回层中的参数'''
        return self._params