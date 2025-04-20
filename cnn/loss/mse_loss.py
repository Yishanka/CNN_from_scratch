from cnn.core import Tensor
from cnn.base.loss import Loss

class MSELoss(Loss):
    def _forward(self, pred: Tensor, true: Tensor) -> Tensor:
        """
        pred: Tensor, shape (batch, num_classes), 是 softmax 后的概率分布
        true: Tensor, shape (batch, 1), 是每个样本的正确类别索引
        """
        batch_size = true.shape[0]
        loss = (((pred - true) ** 2).sum()) / batch_size
        return loss