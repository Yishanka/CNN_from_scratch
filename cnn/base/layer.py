from cnn.core import Tensor, Parameter
class Layer:
    @property
    def parameters(self):
        '''返回层中的参数'''
        return self._params
    
    def __init__(self):
        '''初始化 layer 对象，设置参数列表'''
        self._params: list[Parameter] = []
        self.training = True

    def __setattr__(self, name, value):
        '''收集派生类的所有参数，放到参数列表中'''
        if isinstance(value, Parameter):
            self._params.append(value)
        super().__setattr__(name, value)
    
    def __call__(self, X):
        '''
        标准前向传播函数接口
        Parameters:
            X(Tensor | arraylike): 该层的输入
        Returns:
            Tensor: 该层的输出
        '''
        return self.forward(X)
        
    def forward(self, X)->Tensor:
        '''
        非标准前向传播函数接口，调用 _forward
        Parameters:
            X(Tensor | arraylike): 该层的输入
        Returns:
            Tensor: 该层的输出
        '''
        return self._forward(X)
    
    def zero_grad(self):
        '''将层中参数的梯度归零'''
        for param in self._params:
            param.zero_grad()
    
    def _forward(self, X)->Tensor:
        '''前向传播的抽象函数，需在派生类里实现'''
        raise NotImplementedError('forward 方法未实现')
    
    def train(self):
        '''设置为训练模式'''
        self.training = True
        for param in self._params:
            param.requires_grad = True
        
    def eval(self):
        '''设置为评估模式'''
        self.training = False
        for param in self._params:
            param.requires_grad = False