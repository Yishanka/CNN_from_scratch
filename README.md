# Convolution Neuron Network from Scratch!

## 项目简介
这是一个从零实现的卷积神经网络框架，仅使用 numpy 作为第三方库。支持对图像数据做分类。后续可以扩展出更丰富的功能。

## 设计理念
### 数据设计
项目从零实现了张量类，重载了矩阵的各种运算，并基于链式法则实现单元运算的梯度计算。

### 模型设计
项目以参数为核心，解耦各部分组件：
1. Layer 类定义参数与数据的交互方式，实现前向传播
2. Loss 类计算损失函数，并通过构建计算图实现反向传播
3. Optimizer 定义参数的优化方式，实现梯度下降
4. Model 定义所有操作的接口，每个接口调用对应的组件去操作参数，也可以外部重定义接口

## 项目架构
### 核心组件（项目内部使用）
1. core: 核心组件，包括张量类和派生的参数类，
2. base: 基类包，所有基类（Layer, Loss, Optimizer）都放在这里
### 模型（对外开放接口）
1. layer：模型层包，继承 Layer 类，定义各种神经网络中的层，
2. loss：损失函数包，继承 Loss 类，定义各种损失计算器
3. optimizer: 优化器包，继承 Optimizer 类，定义各种优化器
### 其他
1. data：数据处理包，负责所有和数据处理相关的内容

## 使用示例

```python
import cnn
import cnn
from cnn.data import loader
from cnn.layer import Linear, Conv2d
from cnn.optimizer import Adam
from cnn.loss import CrossEntropyLoss

class Test(cnn.Model):
    def __init__(self):
        super().__init__()
        self.fc = Linear(2,1)
        self.optimizer = Adam()
        self.loss = CrossEntropyLoss()
        
x = []
true = []

test = Test()

pred = test.forward(x)
loss = test.compute_loss(pred, true)
test.backward()
test.step()
test.zero_grad()
```
## 开发细节
1. 具体的模型要都继承基类，保证功能完整
2. 注释尽量规范完整
3. 在对应的包中写好文档，描述开发的进度、理念等
4. 测试人员在根目录写测试文件即可