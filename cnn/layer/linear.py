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
        he_normal(self._weight, fan_in=self._weight.shape[0])  # 使用 He 初始化提高训练初期的稳定性

    def _forward(self, x) -> Tensor:
        '''
        前向传播逻辑：执行线性变换 Y = X @ W + b

        Parameters:
            X (Tensor): 输入张量，形状为 (batch_size, in_features)

        Returns:
            Tensor: 输出张量，形状为 (batch_size, out_features)
        '''
        if not isinstance(x, Tensor):
            x = Tensor(x)

        # 未完成维度检查，可补充
        out = x @ self._weight + self._bias
        return out