from cnn.core import Tensor, Parameter
from cnn.base.loss import Loss

class CrossEntropyLoss(Loss):
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, true):
        return super().forward(pred, true)