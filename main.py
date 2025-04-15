import cnn.nn as nn

class Test(nn.Model):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3,2)
        self.fc2 = nn.Linear(2,1)

test = Test()
params = test.parameters()
for param in params:
    print(param,',')