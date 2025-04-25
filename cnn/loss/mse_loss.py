from cnn.core import Tensor
from cnn.base.loss import Loss

class MSELoss(Loss):
    def __init__(self, lambda1=0, lambda2=0):
        super().__init__(lambda1, lambda2)
    def _forward(self, pred: Tensor, true: Tensor) -> Tensor:
        """
        """
        batch_size = true.shape[0]
        loss = (((pred - true) ** 2).sum()) / batch_size
        return loss