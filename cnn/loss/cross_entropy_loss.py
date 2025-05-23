from cnn.core import Tensor, Parameter
from cnn.base.loss import Loss

class CrossEntropyLoss(Loss):
    def __init__(self, lambda1=0, lambda2=0, reduction='mean'):
        """
        初始化交叉熵损失函数
        
        Parameters:
            reduction (str): 损失缩减方式，'mean'|'sum'|'none'，默认为'mean'
        """
        super().__init__(lambda1, lambda2)
        self.reduction = reduction

    def _forward(self, pred: Tensor, true: Tensor):
        """
        计算交叉熵损失
        
        Parameters:
            pred (Tensor): 预测的概率分布，shape为[batch_size, num_classes]
            true (Tensor): 真实标签，可以是one-hot编码[batch_size, num_classes]或类别索引[batch_size]
            
        Returns:
            Tensor: 计算得到的交叉熵损失
        """
        # 获取批次大小
        batch_size = pred.shape[0]
        
        # 计算交叉熵损失
        eps = 1e-12  # 防止取对数时出现无穷大
        log_probs = (pred + eps).log()
        
        # 根据真实标签的形式选择不同的损失计算方式
        if len(true.shape) == 2:  # one-hot编码
            losses = -(true * log_probs).sum(axis=1)
        else:  # 类别索引
            # 获取每个样本对应类别的预测概率
            true.to_int()
            batch_indices = range(batch_size)
            losses = -log_probs[batch_indices, true]
        
        # 根据reduction方式返回结果
        if self.reduction == 'none':
            return losses
        elif self.reduction == 'sum':
            return losses.sum()
        else:  # 'mean'
            return losses.sum() / batch_size
        
