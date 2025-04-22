import cnn
from cnn.core import Tensor
# from cnn.data import loader
from cnn.layer import Linear, Conv2d, ReLU
from cnn.optimizer import Adam
from cnn.loss import CrossEntropyLoss

class Test(cnn.Model):
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(2,1)
        self.actv1 = ReLU()
        self.optimizer = Adam(lr=1)
        self.loss = CrossEntropyLoss()


x = [10,20]
true = []

test = Test()

pred = test.forward(x)
print(pred)
# loss = test.compute_loss(pred, true)
# test.backward()
# test.step()
# test.zero_grad()

