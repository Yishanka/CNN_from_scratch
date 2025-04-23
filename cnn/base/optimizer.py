from cnn.core import Parameter

class Optimizer:
    def __init__(self):
        pass
    
    def __call__(self, params: list[Parameter]):
        '''
        标准梯度下降接口
        Parameters:
            params(list[Parameter]): 模型的参数
        '''
        self.step(params)

    def step(self, params: list[Parameter]):
        '''
        非标准梯度下降接口
        Parameters:
            params(list[Parameter]): 模型的参数
        '''
        return self._step(params)

    def _step(self, params: list[Parameter]):
        '''梯度下降的抽象函数，需在派生类里实现'''
        raise NotImplementedError("step 方法未实现")