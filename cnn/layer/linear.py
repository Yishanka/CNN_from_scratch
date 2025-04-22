from cnn.core import Parameter, Tensor, he_normal
from cnn.base.layer import Layer

class Linear(Layer):
    def __init__(self, in_features: int, out_features: int):
        '''
        初始化线性变换层。
        Parameters:
            in_features(int): 输入特征的维度
            out_features(int): 输出特征的维度
        '''
        super().__init__()
        self._weight = Parameter((in_features, out_features), is_reg=True)
        self._bias = Parameter((1, out_features), is_reg=False)
        he_normal(self._weight)  # 使用 He 初始化提高训练初期的稳定性

    def _forward(self, X) -> Tensor:
        '''
        前向传播逻辑：执行线性变换 Y = X @ W + b

        Parameters:
            X (Tensor): 输入张量，形状为 (batch_size, in_features)

        Returns:
            Tensor: 输出张量，形状为 (batch_size, out_features)
        '''
        if not isinstance(X, Tensor):
            X = Tensor(X)
        out = X @ self._weight + self._bias
        return out