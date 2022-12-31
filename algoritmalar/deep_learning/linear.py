import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 5)
        self.fc2 = nn.Linear(5, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

net = Net()
print(net)

#create pytorch mse loss example with 2 inputs and 1 output


x = torch.tensor([[2.0], [4.0], [5.0]], requires_grad=True)
y = torch.tensor([[1.0], [2.0], [3.0]], requires_grad=True)

from torch.nn import MSELoss

loss = MSELoss()
output = loss(x, y)
output.backward()

print(output, "\n", x.grad, "\n", y.grad)