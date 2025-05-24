from cnn.core import Tensor, Parameter
from cnn.base import Layer, Loss, Optimizer

class Model:
    '''
    神经网络模型的基础类，定义模型训练的基本流程和接口。
    自动按序收集添加的层、参数、损失函数和优化器，实现 forward、backward 和优化步骤。
    '''
    def __init__(self):
        # 存储模型的所有层（Layer 实例）
        self._layers: list[Layer] = []
        # 当前使用的损失函数（Loss 实例）
        self._loss_fn: Loss = None
        # 当前使用的优化器（Optimizer 实例）
        self._optimizer: Optimizer = None

    @property
    def parameters(self):
        '''属性方法，返回层中的所有参数'''
        params: list[Parameter] = []
        for layer in self._layers:
            params.extend(layer.parameters)
        return params

    def __setattr__(self, name, value):
        '''
        属性赋值时自动检测组件类型：
        - 如果是 Layer，则添加到 _layers 并提取参数。
        - 如果是 Loss 或 Optimizer，则保存为内部属性。
        '''
        if isinstance(value, Layer):
            self._layers.append(value)
        elif isinstance(value, Loss):
            object.__setattr__(self, "_loss", value)
        elif isinstance(value, Optimizer):
            object.__setattr__(self, "_optimizer", value)
        object.__setattr__(self, name, value)
    
    def sequential(self, *layers):
        '''初始化或增加新的layers'''
        layers = list(layers)
        for layer in layers:
            self._layers.append(layer)

    def compile(self, loss: Loss, optimizer: Optimizer):
        '''初始化或增加新的优化器与损失'''
        self._loss_fn = loss
        self._optimizer = optimizer
    
    def forward(self, x) -> Tensor:
        '''
        执行前向传播，依次通过模型中所有层。
        Parameters:
            x : 输入，可以是 numpy 数组，也可以是张量
        Returns:
            Tensor: 模型预测输出
        '''
        for layer in self._layers:
            x = layer(x)
        return x

    def loss(self, pred, true) -> Tensor:
        '''
        计算损失函数
        Parameters:
            pred: 模型预测值
            true: 真实标签
        Returns:
            Tensor: 损失值
        '''
        return self._loss_fn(pred, true, self.parameters)

    def backward(self, remove_graph=True):
        '''通过保存的 loss 执行反向传播'''
        self._loss_fn.backward(remove_graph)

    def step(self):
        '''使用优化器更新所有参数'''
        self._optimizer(self.parameters)

    def zero_grad(self):
        '''将所有参数的梯度清零'''
        for layer in self._layers:
            layer.zero_grad()

    # def fit(self, epoch):
    #     '''继承模型训练方法'''
    #     pass

    def train(self):
        '''设置模型为训练模式'''
        for layer in self._layers:
            layer.train()
    
    def eval(self):
        '''设置模型为测试模式'''
        for layer in self._layers:
            layer.eval()