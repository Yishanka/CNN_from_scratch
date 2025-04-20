from cnn.core import Tensor

class Loss:
    def __init__(self):
        self._loss: Tensor = None
        
    def __repr__(self):
        return self._loss
    
    def __call__(self, pred, true) -> Tensor:
        '''
        标准损失计算接口
        Parameters:
            pred(Tensor): 模型的预测值
            true(Tensor|array_like): 数据集的真实值
        Returns:
            Tensor: 预测值和真实值的损失
        '''
        return self.forward(pred, true)
    
    def forward(self, pred, true) -> Tensor:
        '''
        非标准损失计算接口
        Parameters:
            pred(Tensor): 模型的预测值
            true(Tensor|array_like): 数据集的真实值
        Returns:
            Tensor: 预测值和真实值的损失
        '''
        self._loss = self._forward(pred, true)
        return self._loss
    
    def backward(self, retain_graph=False):
        '''
        反向传播接口
        Parameters:
            retain_graph(bool): 是否保存计算图
        '''
        assert isinstance(self._loss, Tensor), "loss 尚未计算，不能反向传播"
        assert self._loss.size == 1, "只能对标量调用 backward"
        self._loss.backward(retain_graph)

    def _forward(self, pred, true) -> Tensor:
        '''损失计算的抽象函数，需在派生类里实现'''
        raise NotImplementedError("forward 方法未实现")