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