### 对model实现了两种初始化的架构

#### 两种架构的区别
##### 动态初始化

```python
import cnn
from cnn.layer import Layer
class SimpleCNN(cnn.Model)
  def __init__(self):
      super().__init__()
      self.layer1 = Layer(...)
      self.layer2 = Layer(...)
model = SimpleCNN()
```

##### 静态初始化
```python
import cnn
from cnn.layer import Layer
model = cnn.Model()
model.sequential(Layer(), Layer(), ...)
model.compile(Loss(), Optimizer())
```