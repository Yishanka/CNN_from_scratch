from cnn.core import Tensor

class Loss:
    def __repr__(self):
        return self._loss
    
    def __call__(self, pred, true):
        ''' 调用时直接执行 forward，保存 loss: Tensor 实例 '''
        self._loss = self.forward(pred, true)
        return self._loss

    def forward(self, pred, true) -> Tensor:
        raise NotImplementedError("forward 方法未实现")

    def backward(self, retain_graph=False):
        ''' 基于保存的 self.loss 进行反向传播 '''
        assert isinstance(self._loss, Tensor), "loss 尚未计算，不能反向传播"
        assert self._loss.size == 1, "只能对标量调用 backward"
        self._loss.backward(retain_graph)