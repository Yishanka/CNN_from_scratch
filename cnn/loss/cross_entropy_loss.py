from cnn.core import Tensor
from cnn.base.loss import Loss

class CrossEntropyLoss(Loss):
    def _forward(self, pred: Tensor, true: Tensor) -> Tensor:
        """
        pred: Tensor, shape (batch, num_classes), 是 softmax 后的概率分布
        true: Tensor, shape (batch,), 是每个样本的正确类别索引
        """
        batch_size = true.shape[0]
        log_prob = pred.log()
        # 选出每个样本的正确类的 log prob
        correct_log_prob = log_prob[range(batch_size), true]
        loss = -correct_log_prob.sum() / batch_size
        return loss