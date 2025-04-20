from cnn.core import Tensor, Parameter
class Loss:
    def __init__(self, lambda1=0, lambda2=0):
        self._loss: Tensor = None
        self._lambda1 = lambda1
        self._lambda2 = lambda2

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
    
    def forward(self, pred, true, params:list[Parameter]) -> Tensor:
        '''
        非标准损失计算接口
        Parameters:
            pred(Tensor): 模型的预测值
            true(Tensor|array_like): 数据集的真实值
        Returns:
            Tensor: 预测值和真实值的损失
        '''
        loss = self._forward(pred, true)
        # 加上正则项
        l2 = sum([(param ** 2).sum() for param in params])
        l1 = sum([param.abs().sum() for param in params])
        self._loss = loss + self._lambda1 * l1 + self._lambda2 * l2
        return loss
    
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
        '''损失计算的抽象函数，不需要考虑正则化，需在派生类里实现'''
        raise NotImplementedError("forward 方法未实现")