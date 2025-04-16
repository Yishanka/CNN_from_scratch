import cnn
from cnn.data import loader
from cnn.layer import Linear, Conv2d
from cnn.optim import Adam
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

if __name__ == '__main__':
    class A:
        def __init__(self):
            self.a = [1,2,3,4]
        def get(self):
            return self.a

    class B:
        def set(self, a:list):
            self.b = a

        def add(self, x):
            self.b.append(x)

    a = A()
    b = B()
    b.set(a.get())
    b.add(1)
    print(a.get())