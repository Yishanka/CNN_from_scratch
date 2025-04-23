from cnn.core import Tensor
from cnn.base import Layer, Loss, Optimizer
from cnn.base.layer import Layer
from typing import List, Dict, Any, Union, Optional
import numpy as np

class Sequential:
    def __init__(self, *layers):
        """
        Sequential模型，按顺序执行层操作
        
        参数:
            *layers: 网络层实例
        """
        self.layers = layers
        
    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播
        
        参数:
            x: 输入张量
            
        返回:
            模型的输出张量
        """
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def __call__(self, x: Tensor) -> Tensor:
        """调用模型时执行前向传播"""
        return self.forward(x)
    
    def parameters(self):
        """返回模型的所有参数"""
        params = []
        for layer in self.layers:
            if hasattr(layer, 'parameters'):
                params.extend(layer.parameters())
        return params
    
    def train(self):
        """设置为训练模式"""
        for layer in self.layers:
            if hasattr(layer, 'train'):
                layer.train()
    
    def eval(self):
        """设置为评估模式"""
        for layer in self.layers:
            if hasattr(layer, 'eval'):
                layer.eval()
    
    def zero_grad(self):
        """清除所有参数的梯度"""
        for layer in self.layers:
            if hasattr(layer, 'zero_grad'):
                layer.zero_grad()
    
    def __repr__(self):
        layer_reprs = '\n  '.join([repr(layer) for layer in self.layers])
        return f"Sequential(\n  {layer_reprs}\n)"


class Module(Layer):
    """
    自定义神经网络模块的基类
    """
    def __init__(self):
        super().__init__()
        self._modules = {}
        
    def __setattr__(self, name, value):
        if isinstance(value, (Layer, Module)):
            self._modules[name] = value
        super().__setattr__(name, value)
        
    def parameters(self):
        """返回模块的所有参数"""
        params = super().parameters()
        for module in self._modules.values():
            if hasattr(module, 'parameters'):
                params.extend(module.parameters())
        return params
    
    def train(self):
        """设置为训练模式"""
        for module in self._modules.values():
            if hasattr(module, 'train'):
                module.train()
    
    def eval(self):
        """设置为评估模式"""
        for module in self._modules.values():
            if hasattr(module, 'eval'):
                module.eval()
    
    def zero_grad(self):
        """清除所有参数的梯度"""
        super().zero_grad()
        for module in self._modules.values():
            if hasattr(module, 'zero_grad'):
                module.zero_grad()
                
    def __call__(self, x):
        """调用模型时执行前向传播"""
        return self.forward(x)


# 定义一个CNN模型类
class SimpleCNN(Module):
    def __init__(self, in_channels=1, num_classes=10, input_size=28):
        """
        简单的卷积神经网络模型
        
        参数:
            in_channels: 输入通道数，默认为1（灰度图像）
            num_classes: 输出类别数，默认为10
            input_size: 输入图像的尺寸，默认为28x28
        """
        super().__init__()
        from cnn.layer.convolution import Conv2d
        from cnn.layer.pooling import MaxPool2d
        from cnn.layer.flatten import Flatten
        from cnn.layer.linear import Linear
        from cnn.layer.activation import ReLU
        
        # 计算卷积后的特征图大小
        feature_size = input_size // 4  # 经过两次池化，尺寸缩小为原来的1/4
        
        # 简化的网络结构
        self.features = Sequential(
            # 第一个卷积块
            Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1),
            ReLU(),
            MaxPool2d(kernel_size=2),
            
            # 第二个卷积块
            Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            ReLU(),
            MaxPool2d(kernel_size=2)
        )
        
        # 分类器
        self.classifier = Sequential(
            Flatten(),
            Linear(32 * feature_size * feature_size, 64),  # 减少中间层神经元数量
            ReLU(),
            Linear(64, num_classes)
        )
    
    def forward(self, x: Tensor) -> Tensor:
        """前向传播"""
        x = self.features(x)
        x = self.classifier(x)
        return x


class Model:
    def __init__(self, network=None, loss=None, optimizer=None):
        """
        模型类，整合网络、损失函数和优化器
        
        参数:
            network: 神经网络
            loss: 损失函数
            optimizer: 优化器
        """
        self.network = network
        self.loss_fn = loss
        self.optimizer = optimizer
        
    def forward(self, x) -> Tensor:
        """执行前向传播"""
        return self.network(x) if self.network else x
    
    def compute_loss(self, pred, true) -> Tensor:
        """计算损失"""
        if not self.loss_fn:
            raise ValueError("No loss function specified")
        return self.loss_fn(pred, true)
    
    def backward(self):
        """执行反向传播"""
        if self.loss_fn:
            self.loss_fn.backward()
    
    def step(self):
        """更新参数"""
        if self.optimizer:
            self.optimizer.step()
    
    def zero_grad(self):
        """清除梯度"""
        if self.network:
            if hasattr(self.network, 'zero_grad'):
                self.network.zero_grad()

if __name__ == '__main__':
    pass